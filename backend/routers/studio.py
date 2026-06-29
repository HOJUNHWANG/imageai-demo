"""Profile-driven generation, instruction editing and runtime endpoints."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

from ..core.config import (
    ATTENTION_BACKEND,
    DEFAULT_PROFILE,
    EDIT_PROFILES,
    GENERATE_PROFILES,
    HF_TOKEN,
    MAX_UPLOAD_MB,
    get_profile,
)
from ..core.images import (
    composite_at_original_resolution,
    decode_image,
    decode_mask,
    encode_png,
    prepare_for_model,
)
from ..core.models import MODELS
from ..core.runtime import INFERENCE_LOCK, JOB, clear_memory, hardware_info

router = APIRouter()
logger = logging.getLogger(__name__)
MAX_UPLOAD = MAX_UPLOAD_MB * 1024 * 1024
ProfileId = Literal["quality", "balanced", "fast"]


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    profile: ProfileId = DEFAULT_PROFILE  # type: ignore[assignment]
    width: int = Field(1024, ge=512, le=1536)
    height: int = Field(1024, ge=512, le=1536)
    seed: int = Field(-1, ge=-1, le=2**31 - 1)


def _seed(value: int) -> int:
    return value if value >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())


def _fit_dimensions(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    width = max(512, round(width / 32) * 32)
    height = max(512, round(height / 32) * 32)
    if width * height > max_pixels:
        scale = (max_pixels / (width * height)) ** 0.5
        width = max(512, round(width * scale / 32) * 32)
        height = max(512, round(height * scale / 32) * 32)
    return width, height


def _timings(load: float, inference: float, postprocess: float, total: float) -> dict:
    return {
        "load": round(load, 2),
        "inference": round(inference, 2),
        "postprocess": round(postprocess, 2),
        "total": round(total, 2),
    }


def _job_failure_message(exc: Exception) -> str:
    message = str(exc)
    lowered = message.lower()
    if message == "CANCELLED_BY_USER":
        return "Cancelled"
    if "out of memory" in lowered or isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError)):
        return "GPU out of memory"
    if message.startswith("INSUFFICIENT_DISK:"):
        return "Not enough disk space for the selected model"
    if "gated" in lowered or "repository access" in lowered or "401" in message or "403" in message:
        return "Hugging Face model access is required"
    return "The local model failed"


def _run_generate(request: GenerateRequest) -> dict:
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("GPU_BUSY")
    started = time.perf_counter()
    profile = get_profile("generate", request.profile)
    JOB.begin("generate", f"Preparing {profile.label} generation")
    try:
        pipe, load_elapsed, warm_model = MODELS.get("generate", request.profile)
        JOB.check_cancelled()
        JOB.update("preparing", "Preparing dimensions, prompt and seed", stage_progress=0.35)
        seed = _seed(request.seed)
        width, height = _fit_dimensions(request.width, request.height, profile.max_pixels)
        JOB.update("inference", "Encoding prompt and creating latents", 0, profile.steps)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=request.prompt.strip(),
                width=width,
                height=height,
                num_inference_steps=profile.steps,
                guidance_scale=profile.guidance,
                max_sequence_length=512,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=JOB.callback(profile.steps),
            ).images[0]
        inference_elapsed = time.perf_counter() - inference_started
        JOB.update("encoding", "Encoding final PNG", stage_progress=0.25)
        post_started = time.perf_counter()
        encoded = encode_png(image)
        JOB.update("encoding", "Finalizing result", stage_progress=0.9)
        post_elapsed = time.perf_counter() - post_started
        elapsed = time.perf_counter() - started
        JOB.finish(f"{profile.label} generated in {elapsed:.1f}s")
        return {
            "image": encoded,
            "seed": seed,
            "elapsed": round(elapsed, 2),
            "width": width,
            "height": height,
            "profile": profile.id,
            "model": profile.model_id,
            "warm_model": warm_model,
            "timings": _timings(load_elapsed, inference_elapsed, post_elapsed, elapsed),
        }
    except Exception as exc:
        JOB.fail(_job_failure_message(exc))
        raise
    finally:
        INFERENCE_LOCK.release()


def _run_edit(
    original: Image.Image,
    mask: Image.Image | None,
    prompt: str,
    negative: str,
    profile_id: str,
    seed_value: int,
    feather: int,
) -> dict:
    if not INFERENCE_LOCK.acquire(blocking=False):
        raise RuntimeError("GPU_BUSY")
    started = time.perf_counter()
    profile = get_profile("edit", profile_id)
    JOB.begin("edit", f"Preparing {profile.label} edit")
    try:
        pipe, load_elapsed, warm_model = MODELS.get("edit", profile_id)
        JOB.check_cancelled()
        JOB.update("preparing", "Resizing source image for the edit model", stage_progress=0.25)
        model_image = prepare_for_model(original, profile.long_side, profile.max_pixels)
        seed = _seed(seed_value)
        JOB.update("preparing", "Preparing instruction, image and seed", stage_progress=0.85)
        JOB.update("inference", "Encoding source image and instruction", 0, profile.steps)
        inference_started = time.perf_counter()
        with torch.inference_mode():
            edited = pipe(
                image=[model_image],
                prompt=prompt.strip(),
                negative_prompt=negative.strip() or " ",
                num_inference_steps=profile.steps,
                true_cfg_scale=profile.true_cfg,
                guidance_scale=profile.guidance,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=JOB.callback(profile.steps),
            ).images[0]
        inference_elapsed = time.perf_counter() - inference_started
        JOB.update("composite", "Restoring original resolution", stage_progress=0.2)
        post_started = time.perf_counter()
        output = composite_at_original_resolution(original, edited, mask, feather)
        JOB.update("encoding", "Encoding final PNG", stage_progress=0.25)
        encoded = encode_png(output)
        JOB.update("encoding", "Finalizing result", stage_progress=0.9)
        post_elapsed = time.perf_counter() - post_started
        elapsed = time.perf_counter() - started
        JOB.finish(f"{profile.label} edit completed in {elapsed:.1f}s")
        return {
            "image": encoded,
            "seed": seed,
            "elapsed": round(elapsed, 2),
            "width": output.width,
            "height": output.height,
            "masked": mask is not None,
            "profile": profile.id,
            "model": profile.transformer_id or profile.model_id,
            "warm_model": warm_model,
            "timings": _timings(load_elapsed, inference_elapsed, post_elapsed, elapsed),
        }
    except Exception as exc:
        JOB.fail(_job_failure_message(exc))
        raise
    finally:
        INFERENCE_LOCK.release()


def _translate_error(exc: Exception) -> HTTPException:
    if str(exc) == "GPU_BUSY":
        return HTTPException(409, "The GPU is already running another job")
    if str(exc) == "CANCELLED_BY_USER":
        return HTTPException(499, "Cancelled")
    message = str(exc)
    if (
        isinstance(exc, (torch.cuda.OutOfMemoryError, MemoryError))
        or "out of memory" in message.lower()
    ):
        MODELS.unload()
        return HTTPException(507, "GPU out of memory; the active model was unloaded")
    if message.startswith("INSUFFICIENT_DISK:"):
        _, required, free = message.split(":", 2)
        return HTTPException(
            507,
            f"Not enough disk space for this model (need about {required} GB including reserve; {free} GB free)",
        )
    lowered = message.lower()
    if (
        "gated repo" in lowered
        or "gatedrepoerror" in lowered
        or "repository access" in lowered
        or "401" in message
        or "403" in message
    ):
        return HTTPException(424, "This model requires accepting its Hugging Face terms and setting HF_TOKEN")
    if isinstance(exc, (ValueError, UnidentifiedImageError)):
        return HTTPException(422, message or "Invalid image input")
    logger.exception("Studio request failed")
    return HTTPException(500, "The local model failed; check the backend log for details")


async def _read_limited(upload: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await upload.read(1024 * 1024):
        data.extend(chunk)
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, f"File exceeds {MAX_UPLOAD_MB} MB")
    return bytes(data)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/status")
def status():
    return {"job": JOB.snapshot(), "model": MODELS.status, "hardware": hardware_info()}


@router.get("/config")
def config():
    try:
        from huggingface_hub import get_token

        token_configured = bool(HF_TOKEN or get_token())
    except Exception:
        token_configured = bool(HF_TOKEN)
    return {
        "default_profile": DEFAULT_PROFILE,
        "profiles": {
            "generate": {
                key: {**value.public(), "cached": MODELS.is_profile_cached(value)}
                for key, value in GENERATE_PROFILES.items()
            },
            "edit": {
                key: {**value.public(), "cached": MODELS.is_profile_cached(value)}
                for key, value in EDIT_PROFILES.items()
            },
        },
        "hf_token_configured": token_configured,
        "free_disk_gb": MODELS.free_disk_gb(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "safety_checker": False,
        "attention_backend": ATTENTION_BACKEND,
        "note": "No application-level or Diffusers pipeline safety checker is installed.",
    }


@router.post("/generate")
async def generate(request: GenerateRequest):
    try:
        return await asyncio.to_thread(_run_generate, request)
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/edit")
async def edit(
    image: UploadFile = File(...),
    prompt: str = Form(..., min_length=1, max_length=4000),
    negative: str = Form("", max_length=4000),
    mask: UploadFile | None = File(None),
    profile: ProfileId = Form(DEFAULT_PROFILE),  # type: ignore[assignment]
    seed: int = Form(-1, ge=-1, le=2**31 - 1),
    feather: int = Form(8, ge=0, le=64),
):
    if not prompt.strip():
        raise HTTPException(422, "Prompt is required")
    try:
        image_bytes = await _read_limited(image)
        original = decode_image(image_bytes)
        final_mask = None
        if mask is not None:
            mask_bytes = await _read_limited(mask)
            if mask_bytes:
                final_mask = decode_mask(mask_bytes, original.size)
        return await asyncio.to_thread(
            _run_edit,
            original,
            final_mask,
            prompt,
            negative,
            profile,
            seed,
            feather,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise _translate_error(exc) from exc


@router.post("/cancel")
def cancel():
    snapshot = JOB.snapshot()
    if not snapshot["active"]:
        raise HTTPException(409, "No active job to cancel")
    if not JOB.cancel():
        raise HTTPException(409, "Model download and loading cannot be interrupted safely")
    return {"status": "cancellation_requested"}


@router.post("/unload")
def unload():
    if JOB.snapshot()["active"]:
        raise HTTPException(409, "Cannot unload while inference is active")
    MODELS.unload()
    clear_memory()
    return {"status": "unloaded", "hardware": hardware_info()}
