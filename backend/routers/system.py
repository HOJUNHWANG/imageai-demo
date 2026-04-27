"""
System router: VRAM monitoring, cleanup, and inference progress tracking.
"""
import time
from fastapi import APIRouter
from ..core.vram import get_vram_info, get_vram_text, soft_clear_vram
from ..core.pipeline import hard_clear_vram, unload_aux_pipelines

router = APIRouter()

# ─── Global progress state ───
PROGRESS = {
    "active": False,
    "task": "",          # "generate" or "edit"
    "step": 0,
    "total": 0,
    "status": "idle",    # "loading_model", "running", "encoding", "done", "error"
    "message": "",
    "started_at": 0,
    "elapsed": 0,
}


def reset_progress():
    PROGRESS["active"] = False
    PROGRESS["step"] = 0
    PROGRESS["total"] = 0
    PROGRESS["status"] = "idle"
    PROGRESS["message"] = ""
    PROGRESS["elapsed"] = 0
    PROGRESS["started_at"] = 0


def set_progress(task: str, status: str, message: str, step: int = 0, total: int = 0):
    PROGRESS["active"] = status not in ("idle", "done", "error")
    PROGRESS["task"] = task
    PROGRESS["step"] = step
    PROGRESS["total"] = total
    PROGRESS["status"] = status
    PROGRESS["message"] = message
    if status == "loading_model":
        PROGRESS["started_at"] = time.time()
    if PROGRESS["started_at"] > 0:
        PROGRESS["elapsed"] = round(time.time() - PROGRESS["started_at"], 1)

CANCEL_FLAG = False


def request_cancel():
    global CANCEL_FLAG
    CANCEL_FLAG = True
    print("[CANCEL] Cancel requested by user")


def clear_cancel():
    global CANCEL_FLAG
    CANCEL_FLAG = False


def check_cancel(task: str = "", pipe=None):
    """Check if cancel was requested. Call between pipeline phases."""
    if CANCEL_FLAG:
        if task:
            set_progress(task, "error", "Cancelled by user")
        if pipe and hasattr(pipe, "_interrupt"):
            pipe._interrupt = True
        print(f"[CANCEL] Cancel detected in {task or 'unknown'}")
        raise RuntimeError("CANCELLED_BY_USER")


def make_step_callback(task: str, total_steps: int):
    """Create a diffusers pipeline callback that updates progress and checks cancel."""
    def callback(pipe, step, timestep, kwargs):
        if CANCEL_FLAG:
            print(f"[CANCEL] Cancel detected at step {step + 1}/{total_steps}")
            set_progress(task, "error", "Cancelled by user")
            if hasattr(pipe, "_interrupt"):
                pipe._interrupt = True
            raise RuntimeError("CANCELLED_BY_USER")
        current = step + 1
        set_progress(task, "running", f"Step {current}/{total_steps}", current, total_steps)
        # After the last diffusion step, VAE decode begins. Notify the user so
        # the UI doesn't appear frozen during what can be a 10-60s decode pass.
        if current == total_steps:
            set_progress(task, "encoding", "Decoding image (VAE)...")
        return kwargs
    return callback


@router.post("/cancel")
async def cancel_inference():
    """Request cancellation of the current inference."""
    request_cancel()
    return {"status": "cancel_requested"}


@router.get("/progress")
async def get_progress():
    """Return current inference progress."""
    if PROGRESS["started_at"] > 0 and PROGRESS["active"]:
        PROGRESS["elapsed"] = round(time.time() - PROGRESS["started_at"], 1)
    return PROGRESS


@router.get("/vram")
async def vram_status():
    # fast=True: skip synchronize() so polling never blocks active inference
    return get_vram_info(fast=True)


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/clear/soft")
async def clear_soft():
    return soft_clear_vram()


@router.post("/clear/hard")
async def clear_hard():
    result = hard_clear_vram()
    return {**result, "vram": get_vram_info()}


@router.post("/clear/aux")
async def clear_aux():
    unload_aux_pipelines()
    return {"message": "Auxiliary pipelines unloaded.", "vram": get_vram_info()}
