"""
Kontext router: FLUX.1-Kontext text-guided image editing.
No mask needed — edits based on natural language instructions while preserving the original.
"""
import io
import time
import asyncio
import torch
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image

from ..core.pipeline import get_kontext_pipe, hard_clear_vram
from ..core.utils import pil_to_base64
from ..core.vram import get_vram_info
from .system import set_progress, make_step_callback, clear_cancel, check_cancel

router = APIRouter()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    w, h = pil.size
    scale = long_side / max(w, h)
    nw = max(64, (int(w * scale) // 64) * 64)
    nh = max(64, (int(h * scale) // 64) * 64)
    return pil.resize((nw, nh), Image.LANCZOS)


def _run_kontext(pil_image, prompt, steps, guidance, seed):
    clear_cancel()

    set_progress("edit", "loading_model", "Loading FLUX Kontext pipeline...")
    pipe = get_kontext_pipe()
    if pipe is None:
        set_progress("edit", "error", "Failed to load FLUX Kontext pipeline")
        return {"error": "Failed to load FLUX Kontext pipeline.", "status": "error"}

    check_cancel("edit")

    set_progress("edit", "preprocessing", "Preparing image...")

    s = seed
    if s < 0:
        s = int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(s)

    check_cancel("edit")

    set_progress("edit", "running", f"FLUX Kontext (0/{steps})", 0, steps)
    callback = make_step_callback("edit", steps)

    start = time.time()
    with torch.inference_mode():
        result_image = pipe(
            image=pil_image,
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            max_sequence_length=512,
            callback_on_step_end=callback,
        ).images[0]

    set_progress("edit", "encoding", "Encoding result...")
    elapsed = round(time.time() - start, 2)
    set_progress("edit", "done", f"Done in {elapsed}s")

    return {
        "image": pil_to_base64(result_image),
        "seed": s,
        "elapsed": elapsed,
        "status": "success",
        "vram": get_vram_info(),
    }


@router.post("/kontext")
async def kontext_edit(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    steps: int = Form(28),
    guidance: float = Form(2.5),
    seed: int = Form(-1),
    working_long_side: int = Form(1024),
):
    """FLUX Kontext text-guided editing — no mask required."""
    _MAX_UPLOAD = 25 * 1024 * 1024
    try:
        img_bytes = await image.read()
        if len(img_bytes) > _MAX_UPLOAD:
            return {"error": "Image too large (max 25 MB).", "status": "error"}
        pil_image = _bytes_to_pil(img_bytes)
        del img_bytes

        pil_image = _resize_to_long_side(pil_image, working_long_side)

        result = await asyncio.to_thread(
            _run_kontext, pil_image, prompt, steps, guidance, seed
        )
        return result
    except RuntimeError as e:
        if "CANCELLED_BY_USER" in str(e):
            return {"error": "Cancelled by user.", "status": "cancelled"}
        set_progress("edit", "error", str(e))
        return {"error": str(e), "status": "error"}
    except torch.cuda.OutOfMemoryError:
        hard_clear_vram()
        set_progress("edit", "error", "Out of Memory")
        return {"error": "Out of Memory. VRAM cleared.", "status": "oom"}
    except Exception as e:
        set_progress("edit", "error", str(e))
        return {"error": str(e), "status": "error"}
