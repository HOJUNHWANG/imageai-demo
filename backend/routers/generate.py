"""
Generate router: FLUX text-to-image with real-time progress tracking.
Inference runs in a background thread so /api/progress stays responsive.
"""
import time
import asyncio
import torch
from fastapi import APIRouter
from pydantic import BaseModel, Field
from PIL import Image

from ..core.pipeline import get_txt2img_pipe, switch_mode
from ..core.vram import get_vram_info
from ..core.utils import pil_to_base64
from ..core.config import DEVICE
from .system import set_progress, reset_progress, make_step_callback, clear_cancel, check_cancel

router = APIRouter()


class GenerateRequest(BaseModel):
    prompt: str
    width: int = Field(1024, ge=256, le=1536)
    height: int = Field(1024, ge=256, le=1536)
    steps: int = Field(4, ge=1, le=12)
    seed: int = -1




def _run_generate(prompt, width, height, steps, seed):
    """Blocking inference — runs in a thread so async event loop stays free."""
    from ..core.pipeline import get_txt2img_pipe, switch_mode, CURRENT_MODE as cm
    clear_cancel()

    # Phase 1: Model loading
    set_progress("generate", "loading_model", "Loading FLUX pipeline...")

    if cm != "generate":
        switch_mode("generate")

    check_cancel("generate")  # Cancel check after model load

    flux_pipe = get_txt2img_pipe()
    if flux_pipe is None:
        set_progress("generate", "error", "Failed to load FLUX pipeline")
        return {"error": "Failed to load FLUX pipeline.", "status": "error"}

    # Seed
    s = seed
    if s < 0:
        s = int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(s)

    # FLUX VAE uses 16x downscale — align dimensions to prevent padding artifacts
    width = max(256, (width // 16) * 16)
    height = max(256, (height // 16) * 16)

    # Phase 2: Inference with step callback
    set_progress("generate", "running", f"Starting generation (0/{steps})", 0, steps)
    callback = make_step_callback("generate", steps)

    start = time.time()
    with torch.inference_mode():
        result = flux_pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            generator=gen,
            guidance_scale=0.0,
            max_sequence_length=512,
            callback_on_step_end=callback,
        )

    # Phase 3: Encoding
    set_progress("generate", "encoding", "Encoding image...")
    image = result.images[0]
    elapsed = round(time.time() - start, 2)

    set_progress("generate", "done", f"Done in {elapsed}s")

    return {
        "image": pil_to_base64(image),
        "seed": s,
        "elapsed": elapsed,
        "status": "success",
        "vram": get_vram_info(),
    }


@router.post("/generate")
async def generate_image(req: GenerateRequest):
    """Generate an image — runs inference in a thread so progress polling works."""
    try:
        result = await asyncio.to_thread(
            _run_generate, req.prompt, req.width, req.height, req.steps, req.seed
        )
        return result
    except RuntimeError as e:
        if "CANCELLED_BY_USER" in str(e):
            return {"error": "Cancelled by user.", "status": "cancelled"}
        set_progress("generate", "error", str(e))
        return {"error": str(e), "status": "error"}
    except torch.cuda.OutOfMemoryError:
        from ..core.pipeline import hard_clear_vram
        hard_clear_vram()
        set_progress("generate", "error", "Out of Memory")
        return {"error": "Out of Memory. VRAM cleared — try smaller dimensions.", "status": "oom"}
    except Exception as e:
        set_progress("generate", "error", str(e))
        return {"error": str(e), "status": "error"}
