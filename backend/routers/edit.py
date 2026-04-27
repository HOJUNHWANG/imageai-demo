"""
Edit router: SDXL inpainting with real-time progress tracking.
Supports optional ControlNet (canny/depth/openpose).
Inference runs in a background thread so /api/progress stays responsive.
"""
import io
import time
import asyncio
import torch
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image, ImageFilter
from typing import Literal

from ..core.pipeline import (
    get_inpaint_pipe,
    controlnet_pipes, unload_aux_pipelines, hard_clear_vram,
    cv2, _apply_optimizations, _aggressive_vram_cleanup,
)
from ..core.config import (
    DEVICE, AUTO_UNLOAD_AUX, AUTO_HARD_CLEAR_THRESHOLD,
    CONTROLNET_DEPTH, CONTROLNET_OPENPOSE, CONTROLNET_CANNY,
    JUGGERNAUT_INPAINT,
)
from ..core.vram import get_vram_info
from ..core.utils import pil_to_base64
from .system import set_progress, make_step_callback, clear_cancel, check_cancel
from . import system as _system_module

import sys
import os

# Project root must be on sys.path for top-level module imports (prompt_enricher, segformer_masks)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from prompt_enricher import enrich_positive, enrich_negative

router = APIRouter()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


def _resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    w, h = pil.size
    scale = long_side / max(w, h)
    # 64px alignment: SDXL's UNet blocks work in 64px strides, misalignment wastes computation
    nw = max(64, (int(w * scale) // 64) * 64)
    nh = max(64, (int(h * scale) // 64) * 64)
    return pil.resize((nw, nh), Image.LANCZOS)


def _postprocess_mask(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    if cv2 is not None:
        if expand_px > 0:
            # Ellipse kernel gives smoother edges than square; single dilate vs PIL loop
            k = max(3, expand_px * 2 + 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
        if blur_px > 0:
            bk = blur_px if blur_px % 2 == 1 else blur_px + 1
            mask_u8 = cv2.GaussianBlur(mask_u8, (bk, bk), 0)
        return mask_u8
    # PIL fallback (no cv2)
    mask_pil = Image.fromarray(mask_u8).convert("L")
    if expand_px > 0:
        for _ in range(expand_px):
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(3))
    if blur_px > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_px))
    return np.array(mask_pil)


def _preprocess_controlnet_image(image_pil: Image.Image, cn_type: str) -> Image.Image:
    """Preprocess image for ControlNet based on type."""
    if cn_type == "canny":
        if cv2 is not None:
            arr = np.array(image_pil)
            arr = cv2.Canny(arr, 100, 200)
            arr = arr[:, :, None]
            arr = np.concatenate([arr, arr, arr], axis=2)
            return Image.fromarray(arr)
        else:
            # Fallback without cv2: use PIL edge detection
            return image_pil.convert("L").filter(ImageFilter.FIND_EDGES).convert("RGB")
    # depth and openpose: pass the original image directly
    # the ControlNet model handles the preprocessing internally
    return image_pil


def _load_controlnet_pipe(cn_type: str):
    """Load or return cached ControlNet inpaint pipeline.
    Unloads base pipe + other ControlNet types before loading to stay within 12GB VRAM.
    """
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline

    if cn_type in controlnet_pipes:
        return controlnet_pipes[cn_type]

    # Unload other cached ControlNet types
    for other_type in list(controlnet_pipes.keys()):
        if other_type != cn_type:
            controlnet_pipes.pop(other_type, None)

    # Unload all other models to free VRAM before loading CN pipe.
    # Also handles the case where FLUX (txt2img_pipe) is loaded — without this,
    # switching from Generate→ControlNet Edit would load SDXL base first (wasted
    # 20-40 s) only for _load_controlnet_pipe to immediately unload it.
    from ..core import pipeline as _pl_mod
    if _pl_mod.txt2img_pipe is not None:
        print("[CONTROLNET] Unloading FLUX Schnell to free VRAM...")
        _pl_mod.txt2img_pipe = None
    if _pl_mod.fill_pipe is not None:
        print("[CONTROLNET] Unloading FLUX Fill to free VRAM...")
        _pl_mod.fill_pipe = None
    if _pl_mod.kontext_pipe is not None:
        print("[CONTROLNET] Unloading FLUX Kontext to free VRAM...")
        _pl_mod.kontext_pipe = None
    if _pl_mod.pipe is not None:
        print("[CONTROLNET] Unloading base SDXL pipe to free VRAM...")
        _pl_mod.pipe = None
        _pl_mod.PIPE = None
    _pl_mod.CURRENT_MODE = "edit"
    _aggressive_vram_cleanup()

    # Select model repo/path
    if cn_type == "canny":
        repo = CONTROLNET_CANNY
    elif cn_type == "depth":
        repo = CONTROLNET_DEPTH
    elif cn_type == "openpose":
        repo = CONTROLNET_OPENPOSE
    else:
        repo = CONTROLNET_CANNY

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"[CONTROLNET] Loading {cn_type} from {repo}...")
    controlnet = ControlNetModel.from_pretrained(
        repo, torch_dtype=dtype, use_safetensors=True, local_files_only=False
    )

    # Reverted from_pipe because it causes extreme deadlocks with accelerate's CPU offload hooks
    # when _apply_optimizations is called again on the shared modules.
    from ..core.config import JUGGERNAUT_INPAINT, DEFAULT_MODEL
    
    if os.path.exists(JUGGERNAUT_INPAINT):
        cn_pipe = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
            JUGGERNAUT_INPAINT, controlnet=controlnet,
            torch_dtype=dtype, use_safetensors=True,
        )
    else:
        cn_pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            DEFAULT_MODEL, controlnet=controlnet,
            torch_dtype=dtype, safety_checker=None,
        )

    # Force CPU offload for ControlNet: SDXL+ControlNet together exceed 12GB if fully on GPU.
    # CPU offload moves layers to GPU on-demand → ~5-6GB peak instead of ~13-14GB.
    cn_pipe = _apply_optimizations(cn_pipe, f"ControlNet-{cn_type}", force_cpu_offload=True)
    controlnet_pipes[cn_type] = cn_pipe
    print(f"[CONTROLNET] {cn_type} ready on {DEVICE}")
    return cn_pipe


def _encode_prompt(pipe, prompt: str, negative_prompt: str):
    """Single encode_prompt call — SDXL dual CLIP handles 77 tokens each natively.
    With CPU offload, each call moves CLIP-L+CLIP-G (~1.8 GB) CPU↔GPU,
    so a single call is ~2-5s faster than multi-chunk approaches.
    """
    pos_emb, neg_emb, pos_pool, neg_pool = pipe.encode_prompt(
        prompt=prompt or "",
        negative_prompt=negative_prompt or "",
        device=pipe._execution_device,
    )
    return pos_emb, neg_emb, pos_pool, neg_pool


def _run_edit(pil_image, pil_mask, prompt, negative, steps, strength, guidance, seed,
              auto_enrich, mask_expand, mask_blur,
              use_controlnet=False, controlnet_type="canny", cn_scale=0.45,
              protect_face=False, engine="sdxl"):
    """Blocking inference — runs in a thread so async event loop stays free."""
    from ..core.pipeline import get_inpaint_pipe
    clear_cancel()

    # Phase 1: Model loading
    if engine == "flux_fill":
        set_progress("edit", "loading_model", "Loading FLUX Fill pipeline...")
    elif use_controlnet:
        set_progress("edit", "loading_model", f"Preparing ControlNet ({controlnet_type})...")
    else:
        set_progress("edit", "loading_model", "Loading SDXL inpaint pipeline...")
        get_inpaint_pipe()

    check_cancel("edit")  # Cancel check after model load

    # Phase 2: Preprocessing
    set_progress("edit", "preprocessing", "Processing image & mask...")

    mask_u8 = np.array(pil_mask.convert("L"))
    mask_u8 = _postprocess_mask(mask_u8, mask_expand, mask_blur)

    if protect_face:
        # Single SegFormer call extracts face + hair together (was 2 separate load/infer/unload cycles)
        from segformer_masks import segformer_face_hair_mask
        face_hair_mask = segformer_face_hair_mask(pil_image)
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            face_hair_mask = cv2.dilate(face_hair_mask, kernel, iterations=2)
        mask_u8[face_hair_mask > 127] = 0

    mask_pil_final = Image.fromarray(mask_u8).convert("L")

    if auto_enrich:
        enriched_pos = enrich_positive(prompt)
        enriched_neg = enrich_negative(negative)
    else:
        enriched_pos = prompt
        enriched_neg = negative or ""

    s = seed
    if s < 0:
        s = int(torch.randint(0, 2**32 - 1, (1,)).item())
    gen = torch.Generator(device="cpu").manual_seed(s)

    check_cancel("edit")  # Cancel check before inference

    result_image = None

    # FLUX Fill branch
    if engine == "flux_fill":
        from ..core.pipeline import get_fill_pipe
        fill_p = get_fill_pipe()
        if fill_p is None:
            set_progress("edit", "error", "Failed to load FLUX Fill pipeline")
            return {"error": "Failed to load FLUX Fill pipeline.", "status": "error"}

        actual_steps = steps
        set_progress("edit", "running", f"FLUX Fill (0/{actual_steps})", 0, actual_steps)
        callback = make_step_callback("edit", actual_steps)

        # FLUX Fill-dev requires high guidance (30) for coherent inpainting.
        # Override the SDXL default (7.0) which produces blurry results with Fill.
        fill_guidance = 30.0

        start = time.time()
        with torch.inference_mode():
            result_image = fill_p(
                prompt=enriched_pos,
                image=pil_image,
                mask_image=mask_pil_final,
                num_inference_steps=steps,
                guidance_scale=fill_guidance,
                generator=gen,
                max_sequence_length=512,
                callback_on_step_end=callback,
            ).images[0]
        print("[FLUX FILL] Generation success!")

    # SDXL branches (ControlNet or base inpaint) — skip if FLUX Fill already produced a result
    elif engine != "flux_fill":
        # Actual steps run = int(steps * strength) due to diffusers strength scheduling
        actual_steps = max(1, int(steps * strength))

        if use_controlnet:
            set_progress("edit", "loading_model", f"Loading ControlNet ({controlnet_type})...")
            try:
                cn_pipe = _load_controlnet_pipe(controlnet_type)
                control_image = _preprocess_controlnet_image(pil_image, controlnet_type)

                set_progress("edit", "preprocessing", f"Encoding prompts for ControlNet ({controlnet_type})...")

                # Encode prompt
                pos_emb, neg_emb, pos_pool, neg_pool = _encode_prompt(cn_pipe, enriched_pos, enriched_neg)

                set_progress("edit", "running", f"ControlNet {controlnet_type} (0/{actual_steps})", 0, actual_steps)
                callback = make_step_callback("edit", actual_steps)

                start = time.time()
                with torch.inference_mode():
                    result_image = cn_pipe(
                        prompt_embeds=pos_emb,
                        negative_prompt_embeds=neg_emb,
                        pooled_prompt_embeds=pos_pool,
                        negative_pooled_prompt_embeds=neg_pool,
                        image=pil_image,
                        mask_image=mask_pil_final,
                        control_image=control_image,
                        controlnet_conditioning_scale=float(cn_scale),
                        num_inference_steps=steps,
                        strength=strength,
                        guidance_scale=guidance,
                        generator=gen,
                        callback_on_step_end=callback,
                    ).images[0]
                print("[CONTROLNET] Generation success!")
            except RuntimeError as e:
                if "CANCELLED_BY_USER" in str(e):
                    raise  # Re-raise cancel
                print(f"[CONTROLNET] Failed: {e} → falling back to base inpaint")
                result_image = None
            except Exception as e:
                print(f"[CONTROLNET] Failed: {e} → falling back to base inpaint")
                result_image = None

        # Base inpaint (fallback or no ControlNet)
        if result_image is None:
            # If ControlNet was attempted and failed, its pipe (~8-10 GB) is still
            # cached. Loading base SDXL (~6 GB) on top would exceed 12 GB → OOM.
            # Clear the CN cache and free VRAM before proceeding.
            if use_controlnet and controlnet_pipes:
                controlnet_pipes.clear()
                _aggressive_vram_cleanup()
                set_progress("edit", "loading_model", "Loading base SDXL (ControlNet fallback)...")

            p = get_inpaint_pipe()
            if p is None:
                set_progress("edit", "error", "Failed to load pipeline")
                return {"error": "Failed to load inpaint pipeline.", "status": "error"}

            set_progress("edit", "preprocessing", f"Encoding prompts ({actual_steps} steps)...")

            # Encode prompt
            pos_emb, neg_emb, pos_pool, neg_pool = _encode_prompt(p, enriched_pos, enriched_neg)

            set_progress("edit", "running", f"Starting inpaint (0/{actual_steps})", 0, actual_steps)

            callback = make_step_callback("edit", actual_steps)

            start = time.time()
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

    # Hard composite: force pixel-perfect original preservation outside the mask.
    # The inpaint model may bleed minor changes at mask boundaries; this ensures
    # anything outside the mask is the exact original pixel.
    if result_image is not None:
        if result_image.size != pil_image.size:
            result_image = result_image.resize(pil_image.size, Image.LANCZOS)
        result_image = Image.composite(
            result_image.convert("RGB"),
            pil_image.convert("RGB"),
            mask_pil_final,
        )

    # Phase 4: Encoding
    set_progress("edit", "encoding", "Encoding result...")
    elapsed = round(time.time() - start, 2)

    if AUTO_UNLOAD_AUX:
        unload_aux_pipelines()

    set_progress("edit", "done", f"Done in {elapsed}s")

    return {
        "image": pil_to_base64(result_image),
        "seed": s,
        "elapsed": elapsed,
        "prompt_used": enriched_pos,
        "status": "success",
        "vram": get_vram_info(),
    }


@router.post("/edit")
async def edit_image(
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
    use_controlnet: bool = Form(False),
    controlnet_type: Literal["canny", "depth", "openpose"] = Form("canny"),
    cn_scale: float = Form(0.45),
    protect_face: bool = Form(False),
    working_long_side: int = Form(1024),
    engine: Literal["sdxl", "flux_fill"] = Form("sdxl"),
):
    """Apply SDXL inpainting — runs inference in a thread so progress polling works."""
    _MAX_UPLOAD = 25 * 1024 * 1024  # 25 MB
    try:
        # Read files in the async context (non-blocking)
        img_bytes = await image.read()
        if len(img_bytes) > _MAX_UPLOAD:
            return {"error": "Image too large (max 25 MB).", "status": "error"}
        mask_bytes = await mask.read()
        if len(mask_bytes) > _MAX_UPLOAD:
            return {"error": "Mask too large (max 25 MB).", "status": "error"}
        pil_image = _bytes_to_pil(img_bytes)
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        del img_bytes, mask_bytes

        # Resize
        pil_image = _resize_to_long_side(pil_image, working_long_side)
        pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)

        task = asyncio.create_task(asyncio.to_thread(
            _run_edit, pil_image, pil_mask, prompt, negative, steps, strength,
            guidance, seed, auto_enrich, mask_expand, mask_blur,
            use_controlnet, controlnet_type, cn_scale, protect_face, engine
        ))
        while not task.done():
            if _system_module.CANCEL_FLAG:
                return {"error": "Cancelled by user.", "status": "cancelled"}
            await asyncio.sleep(0.25)
            
        result = task.result()
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
