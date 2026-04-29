"""
Test router: experimental model inference (txt2img + inpainting + img2img).
Model identities are opaque — only slot IDs (test_model_1 …) are exposed publicly.
"""
import io
import time
import asyncio
import torch
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image, ImageFilter

from ..core.pipeline import (
    get_test_pipe, get_test_inpaint_pipe, get_test_img2img_pipe, unload_test_pipe,
    hard_clear_vram, cv2, _aggressive_vram_cleanup,
)
from ..core.vram import get_vram_info
from ..core.utils import pil_to_base64
from ..core.config import DEVICE, TEST_MODELS
from .system import set_progress, reset_progress, make_step_callback, clear_cancel, check_cancel

import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompt_enricher import enrich_positive, enrich_negative

router = APIRouter()


# ─── Shared helpers (mirrors edit.py) ────────────────────────────────────────

def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    w, h = pil.size
    scale = long_side / max(w, h)
    nw = max(64, (int(w * scale) // 64) * 64)
    nh = max(64, (int(h * scale) // 64) * 64)
    return pil.resize((nw, nh), Image.LANCZOS)


def _postprocess_mask(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    if cv2 is not None:
        if expand_px > 0:
            k = max(3, expand_px * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        if blur_px > 0:
            bk = blur_px if blur_px % 2 == 1 else blur_px + 1
            mask_u8 = cv2.GaussianBlur(mask_u8, (bk, bk), 0)
        return mask_u8
    mask_pil = Image.fromarray(mask_u8).convert("L")
    if expand_px > 0:
        for _ in range(expand_px):
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    if blur_px > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_px))
    return np.array(mask_pil)


# ─── Generate (txt2img) ───────────────────────────────────────────────────────

from pydantic import BaseModel, Field

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
    set_progress("test", "done", f"Done in {elapsed}s", req.steps, req.steps)
    return {
        "image": pil_to_base64(image),
        "seed": seed,
        "elapsed": elapsed,
        "model_id": req.model_id,
        "status": "ok",
        "vram": get_vram_info(),
    }


# ─── Edit (inpainting) ────────────────────────────────────────────────────────

def _run_test_edit(model_id, pil_image, pil_mask, prompt, negative,
                   steps, strength, guidance, seed,
                   auto_enrich, mask_expand, mask_blur, protect_face):
    clear_cancel()
    set_progress("test", "loading_model", f"Loading {model_id} (inpaint)...")

    p = get_test_inpaint_pipe(model_id)
    if p is None:
        set_progress("test", "error", f"Failed to load {model_id}")
        return {"error": f"Failed to load {model_id}.", "status": "error"}

    check_cancel("test")
    set_progress("test", "preprocessing", "Processing mask...")

    mask_u8 = np.array(pil_mask.convert("L"))
    mask_u8 = _postprocess_mask(mask_u8, mask_expand, mask_blur)

    if protect_face:
        from segformer_masks import segformer_face_hair_mask
        face_mask = segformer_face_hair_mask(pil_image)
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            face_mask = cv2.dilate(face_mask, kernel, iterations=2)
        mask_u8[face_mask > 127] = 0

    mask_pil_final = Image.fromarray(mask_u8).convert("L")

    if auto_enrich:
        enriched_pos = enrich_positive(prompt)
        enriched_neg = enrich_negative(negative)
    else:
        enriched_pos = prompt or ""
        enriched_neg = negative or ""

    s = seed if seed >= 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(s)

    actual_steps = max(1, int(steps * strength))
    set_progress("test", "preprocessing", f"Encoding prompts ({actual_steps} steps)...")

    pos_emb, neg_emb, pos_pool, neg_pool = p.encode_prompt(
        prompt=enriched_pos,
        negative_prompt=enriched_neg,
        device=p._execution_device,
    )

    check_cancel("test")
    set_progress("test", "running", f"Inpainting (0/{actual_steps})", 0, actual_steps)
    callback = make_step_callback("test", actual_steps)

    t0 = time.time()
    try:
        with torch.inference_mode():
            result_image = p(
                prompt_embeds=pos_emb,
                negative_prompt_embeds=neg_emb,
                pooled_prompt_embeds=pos_pool,
                negative_pooled_prompt_embeds=neg_pool,
                image=pil_image,
                mask_image=mask_pil_final,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
                generator=gen,
                callback_on_step_end=callback,
            ).images[0]
    except Exception as e:
        if "cancel" in str(e).lower() or "CANCELLED" in str(e):
            set_progress("test", "cancelled", "Cancelled")
            return {"status": "cancelled"}
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}

    # Pixel-perfect composite: preserve original outside mask
    if result_image.size != pil_image.size:
        result_image = result_image.resize(pil_image.size, Image.LANCZOS)
    result_image = Image.composite(
        result_image.convert("RGB"),
        pil_image.convert("RGB"),
        mask_pil_final,
    )

    elapsed = round(time.time() - t0, 2)
    set_progress("test", "done", f"Done in {elapsed}s")
    return {
        "image": pil_to_base64(result_image),
        "seed": s,
        "elapsed": elapsed,
        "model_id": model_id,
        "status": "ok",
        "vram": get_vram_info(),
    }


# ─── Routes ───────────────────────────────────────────────────────────────────

@router.get("/test/models")
def list_test_models():
    """Return available test model slot IDs. Never reveals actual model paths or names."""
    return {"models": list(TEST_MODELS.keys()), "count": len(TEST_MODELS)}


@router.post("/test/generate")
async def test_generate(req: TestGenerateRequest):
    if req.model_id not in TEST_MODELS:
        return {"error": f"Unknown model slot: {req.model_id}", "status": "error"}
    reset_progress("test")
    result = await asyncio.to_thread(_run_test_generate, req)
    return result


@router.post("/test/edit")
async def test_edit_image(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form(""),
    negative: str = Form(""),
    steps: int = Form(28),
    strength: float = Form(0.55),
    guidance: float = Form(7.0),
    mask_expand: int = Form(10),
    mask_blur: int = Form(18),
    seed: int = Form(-1),
    auto_enrich: bool = Form(True),
    protect_face: bool = Form(False),
    working_long_side: int = Form(1024),
):
    if model_id not in TEST_MODELS:
        return {"error": f"Unknown model slot: {model_id}", "status": "error"}

    _MAX_UPLOAD = 25 * 1024 * 1024
    try:
        img_bytes = await image.read()
        if len(img_bytes) > _MAX_UPLOAD:
            return {"error": "Image too large (max 25 MB).", "status": "error"}
        mask_bytes = await mask.read()
        if len(mask_bytes) > _MAX_UPLOAD:
            return {"error": "Mask too large (max 25 MB).", "status": "error"}

        pil_image = _bytes_to_pil(img_bytes)
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        del img_bytes, mask_bytes

        pil_image = _resize_to_long_side(pil_image, working_long_side)
        pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)

        reset_progress("test")
        result = await asyncio.to_thread(
            _run_test_edit,
            model_id, pil_image, pil_mask, prompt, negative,
            steps, strength, guidance, seed,
            auto_enrich, mask_expand, mask_blur, protect_face,
        )
        return result
    except RuntimeError as e:
        if "CANCELLED_BY_USER" in str(e):
            return {"error": "Cancelled by user.", "status": "cancelled"}
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}
    except torch.cuda.OutOfMemoryError:
        hard_clear_vram()
        set_progress("test", "error", "Out of Memory")
        return {"error": "Out of Memory. VRAM cleared.", "status": "oom"}
    except Exception as e:
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}


@router.post("/test/unload")
def test_unload():
    """Free VRAM occupied by active test pipelines."""
    return unload_test_pipe()


# ─── Img2Img (maskless) ───────────────────────────────────────────────────────

def _run_test_img2img(model_id, pil_image, prompt, negative,
                      steps, strength, guidance, seed, auto_enrich):
    clear_cancel()
    set_progress("test", "loading_model", f"Loading {model_id} (img2img)...")

    p = get_test_img2img_pipe(model_id)
    if p is None:
        set_progress("test", "error", f"Failed to load {model_id}")
        return {"error": f"Failed to load {model_id}.", "status": "error"}

    check_cancel("test")
    set_progress("test", "preprocessing", "Encoding prompts...")

    if auto_enrich:
        enriched_pos = enrich_positive(prompt)
        enriched_neg = enrich_negative(negative)
    else:
        enriched_pos = prompt or ""
        enriched_neg = negative or ""

    s = seed if seed >= 0 else int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(s)

    # img2img actual steps = int(steps * strength)
    actual_steps = max(1, int(steps * strength))

    pos_emb, neg_emb, pos_pool, neg_pool = p.encode_prompt(
        prompt=enriched_pos,
        negative_prompt=enriched_neg,
        device=p._execution_device,
    )

    check_cancel("test")
    set_progress("test", "running", f"Generating (0/{actual_steps})", 0, actual_steps)
    callback = make_step_callback("test", actual_steps)

    t0 = time.time()
    try:
        with torch.inference_mode():
            result_image = p(
                prompt_embeds=pos_emb,
                negative_prompt_embeds=neg_emb,
                pooled_prompt_embeds=pos_pool,
                negative_pooled_prompt_embeds=neg_pool,
                image=pil_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=gen,
                callback_on_step_end=callback,
            ).images[0]
    except Exception as e:
        if "cancel" in str(e).lower() or "CANCELLED" in str(e):
            set_progress("test", "cancelled", "Cancelled")
            return {"status": "cancelled"}
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}

    elapsed = round(time.time() - t0, 2)
    set_progress("test", "done", f"Done in {elapsed}s")
    return {
        "image": pil_to_base64(result_image),
        "seed": s,
        "elapsed": elapsed,
        "model_id": model_id,
        "status": "ok",
        "vram": get_vram_info(),
    }


@router.post("/test/img2img")
async def test_img2img(
    model_id: str = Form(...),
    image: UploadFile = File(...),
    prompt: str = Form(""),
    negative: str = Form(""),
    steps: int = Form(30),
    strength: float = Form(0.6),
    guidance: float = Form(7.0),
    seed: int = Form(-1),
    auto_enrich: bool = Form(True),
    working_long_side: int = Form(1024),
):
    if model_id not in TEST_MODELS:
        return {"error": f"Unknown model slot: {model_id}", "status": "error"}

    _MAX_UPLOAD = 25 * 1024 * 1024
    try:
        img_bytes = await image.read()
        if len(img_bytes) > _MAX_UPLOAD:
            return {"error": "Image too large (max 25 MB).", "status": "error"}

        pil_image = _bytes_to_pil(img_bytes)
        del img_bytes
        pil_image = _resize_to_long_side(pil_image, working_long_side)

        reset_progress("test")
        result = await asyncio.to_thread(
            _run_test_img2img,
            model_id, pil_image, prompt, negative,
            steps, strength, guidance, seed, auto_enrich,
        )
        return result
    except RuntimeError as e:
        if "CANCELLED_BY_USER" in str(e):
            return {"error": "Cancelled by user.", "status": "cancelled"}
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}
    except torch.cuda.OutOfMemoryError:
        hard_clear_vram()
        set_progress("test", "error", "Out of Memory")
        return {"error": "Out of Memory. VRAM cleared.", "status": "oom"}
    except Exception as e:
        set_progress("test", "error", str(e))
        return {"error": str(e), "status": "error"}
