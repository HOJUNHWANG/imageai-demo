"""Inference state, cancellation, progress estimation and hardware telemetry."""
from __future__ import annotations

import gc
import threading
import time
from dataclasses import asdict, dataclass

import psutil
import torch

from .config import DEVICE


STAGE_RANGES = {
    "downloading": (0.0, 10.0),
    "loading": (2.0, 14.0),
    "preparing": (14.0, 18.0),
    "inference": (18.0, 92.0),
    "composite": (92.0, 96.0),
    "encoding": (96.0, 99.5),
    "done": (100.0, 100.0),
}


@dataclass
class JobSnapshot:
    active: bool = False
    task: str = ""
    stage: str = "idle"
    message: str = "Ready"
    step: int = 0
    total: int = 0
    stage_progress: float | None = None
    overall_progress: float = 0.0
    indeterminate: bool = False
    eta_seconds: float | None = None
    started_at: float = 0.0
    stage_started_at: float = 0.0
    stage_elapsed: float = 0.0
    elapsed: float = 0.0
    cancelled: bool = False
    cancellable: bool = False


class JobState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._job = JobSnapshot()

    def begin(self, task: str, message: str) -> None:
        now = time.time()
        with self._lock:
            self._job = JobSnapshot(
                active=True,
                task=task,
                stage="loading",
                message=message,
                overall_progress=2.0,
                indeterminate=True,
                started_at=now,
                stage_started_at=now,
            )

    def update(
        self,
        stage: str,
        message: str,
        step: int = 0,
        total: int = 0,
        stage_progress: float | None = None,
    ) -> None:
        now = time.time()
        with self._lock:
            if stage != self._job.stage:
                self._job.stage = stage
                self._job.stage_started_at = now
            self._job.message = message
            self._job.step = step
            self._job.total = total
            self._job.cancellable = stage == "inference"

            if stage == "inference" and total:
                stage_progress = min(1.0, max(0.0, step / total))
            self._job.stage_progress = stage_progress
            start, end = STAGE_RANGES.get(stage, (0.0, 100.0))
            if stage_progress is None:
                self._job.overall_progress = start
                self._job.indeterminate = stage in {"downloading", "loading"}
            else:
                self._job.overall_progress = start + (end - start) * min(1.0, max(0.0, stage_progress))
                self._job.indeterminate = False

            if stage == "inference" and step > 0:
                elapsed = max(0.001, now - self._job.stage_started_at)
                self._job.eta_seconds = elapsed / step * max(0, total - step)
            else:
                self._job.eta_seconds = None

    def finish(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self._job.active = False
            self._job.stage = "done"
            self._job.message = message
            self._job.stage_progress = 1.0
            self._job.overall_progress = 100.0
            self._job.indeterminate = False
            self._job.cancellable = False
            self._job.eta_seconds = 0.0
            self._job.elapsed = now - self._job.started_at
            self._job.stage_elapsed = now - self._job.stage_started_at

    def fail(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self._job.active = False
            self._job.stage = "error"
            self._job.message = message
            self._job.indeterminate = False
            self._job.cancellable = False
            self._job.eta_seconds = None
            if self._job.started_at:
                self._job.elapsed = now - self._job.started_at
                self._job.stage_elapsed = now - self._job.stage_started_at

    def cancel(self) -> bool:
        with self._lock:
            if not self._job.active or not self._job.cancellable:
                return False
            self._job.cancelled = True
            self._job.message = "Cancellation requested"
            return True

    def check_cancelled(self) -> None:
        with self._lock:
            if self._job.cancelled:
                raise RuntimeError("CANCELLED_BY_USER")

    def callback(self, total: int):
        def on_step(pipe, step, timestep, callback_kwargs):
            self.check_cancelled()
            current = step + 1
            self.update("inference", f"Denoising {current} of {total}", current, total)
            return callback_kwargs

        return on_step

    def snapshot(self) -> dict:
        with self._lock:
            data = asdict(self._job)
            now = time.time()
            if self._job.active and self._job.started_at:
                data["elapsed"] = round(now - self._job.started_at, 1)
                data["stage_elapsed"] = round(now - self._job.stage_started_at, 1)
            else:
                data["elapsed"] = round(self._job.elapsed, 1)
                data["stage_elapsed"] = round(self._job.stage_elapsed, 1)
            if data["eta_seconds"] is not None:
                data["eta_seconds"] = round(data["eta_seconds"], 1)
            data["overall_progress"] = round(data["overall_progress"], 1)
            if data["stage_progress"] is not None:
                data["stage_progress"] = round(data["stage_progress"], 3)
            return data


JOB = JobState()
INFERENCE_LOCK = threading.Lock()


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def hardware_info() -> dict:
    ram = psutil.virtual_memory()
    result = {
        "device": DEVICE,
        "gpu_name": None,
        "vram_total_gb": 0.0,
        "vram_allocated_gb": 0.0,
        "vram_reserved_gb": 0.0,
        "ram_total_gb": round(ram.total / 1024**3, 1),
        "ram_used_gb": round(ram.used / 1024**3, 1),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        result.update(
            gpu_name=torch.cuda.get_device_name(index),
            vram_total_gb=round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 1),
            vram_allocated_gb=round(torch.cuda.memory_allocated(index) / 1024**3, 2),
            vram_reserved_gb=round(torch.cuda.memory_reserved(index) / 1024**3, 2),
        )
    return result
