"""
Test router: experimental model inference (txt2img).
Model identities are opaque — only slot IDs (test_model_1 …) are exposed publicly.
"""
import time
import asyncio
import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..core.pipeline import get_test_pipe, unload_test_pipe
from ..core.vram import get_vram_info
from ..core.utils import pil_to_base64
from ..core.config import DEVICE, TEST_MODELS
from .system import set_progress, reset_progress, make_step_callback, clear_cancel, check_cancel

router = APIRouter()


class TestGenerateRequest(BaseModel):
    model_id: str
    prompt: str
    negative_prompt: str = ""
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)
    steps: int = Field(20, ge=1, le=60)
    guidance: float = Field(7.0, ge=1.0, le=15.0)
    seed: int = -1


def _run_test_generate(req: TestGenerateRequest):
    clear_cancel()
    set_progress("test", "loading_model", f"Loading {req.model_id}...")

    p = get_test_pipe(req.model_id)
    if p is None:
        set_progress("test", "error", f"Failed to load {req.model_id}")
        return {"error": f"Failed to load {req.model_id}.", "status": "error"}

    check_cancel("test")

    seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(seed)

    width = max(256, (req.width // 8) * 8)
    height = max(256, (req.height // 8) * 8)

    set_progress("test", "running", f"Generating (0/{req.steps})", 0, req.steps)
    callback = make_step_callback("test", req.steps)

    t0 = time.time()
    try:
        kwargs: dict = dict(
            prompt=req.prompt,
            width=width,
            height=height,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            generator=gen,
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
        if req.negative_prompt:
            kwargs["negative_prompt"] = req.negative_prompt

        result = p(**kwargs)
    except Exception as e:
        if "cancel" in str(e).lower():
            set_progress("test", "cancelled", "Cancelled")
            return {"status": "cancelled"}
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}

    image = result.images[0]
    elapsed = round(time.time() - t0, 1)
    vram = get_vram_info()

    set_progress("test", "done", f"Done in {elapsed}s", req.steps, req.steps)
    return {
        "image": pil_to_base64(image),
        "seed": seed,
        "elapsed": elapsed,
        "model_id": req.model_id,
        "status": "ok",
        "vram": vram,
    }


@router.get("/test/models")
def list_test_models():
    """Return available test model slot IDs. Never reveals actual model paths or names."""
    return {"models": list(TEST_MODELS.keys()), "count": len(TEST_MODELS)}


@router.post("/test/generate")
async def test_generate(req: TestGenerateRequest):
    if req.model_id not in TEST_MODELS:
        return {"error": f"Unknown model slot: {req.model_id}", "status": "error"}
    reset_progress("test")
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _run_test_generate, req)
    return result


@router.post("/test/unload")
def test_unload():
    """Free VRAM occupied by the active test model."""
    return unload_test_pipe()
