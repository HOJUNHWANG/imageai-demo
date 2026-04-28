"""
Mask router: auto-masking endpoints.
Fixed: pass (pil, helper) to build_*_mask functions.
"""
import io
import logging
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form
from PIL import Image

import sys
import os

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ..core.config import WEIGHTS_DIR

router = APIRouter()

# ─── MPTasksHelper singleton — load once, reuse across requests ───
_mp_helper = None

def _get_mp_helper():
    """Return a cached MPTasksHelper instance (lazy-init, never reloaded)."""
    global _mp_helper
    if _mp_helper is None:
        from mp_tasks_utils import MPTasksHelper
        _mp_helper = MPTasksHelper(weights_dir=WEIGHTS_DIR)
    return _mp_helper


def _pil_to_base64(img) -> str:
    """Encode mask as PNG (lossless — masks need exact pixel values)."""
    from ..core.utils import pil_to_base64
    return pil_to_base64(img, format="PNG")


@router.post("/mask/auto")
async def auto_mask(
    image: UploadFile = File(...),
    target: str = Form("top"),
):
    """Generate auto-mask using MediaPipe. Returns mask candidates."""
    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Use cached MPTasksHelper singleton (no reload overhead)
        mp_helper = _get_mp_helper() if target in ("background", "person") else None

        # Build mask
        if target == "pants":
            from segformer_masks import segformer_bottom_mask
            mask = segformer_bottom_mask(pil)
        elif target == "hair":
            from segformer_masks import segformer_hair_mask
            mask = segformer_hair_mask(pil)
        elif target == "dress":
            from segformer_masks import segformer_dress_mask
            mask = segformer_dress_mask(pil)
        elif target == "face":
            from segformer_masks import segformer_face_mask
            mask = segformer_face_mask(pil)
        elif target == "accessories":
            from segformer_masks import segformer_accessories_mask
            mask = segformer_accessories_mask(pil)
        elif target == "background":
            from mp_tasks_utils import build_background_mask_v5_tasks
            mask = build_background_mask_v5_tasks(pil, mp_helper)
        elif target == "person":
            mask = mp_helper.person_mask(pil)
        else: # target == "top"
            from segformer_masks import segformer_top_mask
            mask = segformer_top_mask(pil)

        masks_b64 = [_pil_to_base64(mask)]

        return {
            "masks": masks_b64,
            "count": len(masks_b64),
            "status": "success",
        }
    except Exception as e:
        logger.exception("[MASK] auto_mask failed")
        return {"error": str(e), "status": "error", "masks": []}


@router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    working_long_side: int = Form(832),
):
    """Upload an image and return resized version."""
    img_bytes = await image.read()
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    w, h = pil.size
    scale = working_long_side / max(w, h)
    # 64px alignment: SDXL UNet operates in 64px strides
    nw = max(64, int(w * scale) // 64 * 64)
    nh = max(64, int(h * scale) // 64 * 64)
    working = pil.resize((nw, nh), Image.LANCZOS)

    return {
        "width": nw,
        "height": nh,
        "image": _pil_to_base64(working),
        "status": "success",
    }


# ─── SAM Click-to-Select Mask ───
_sam_model = None
_sam_processor = None


def _load_sam():
    global _sam_model, _sam_processor
    if _sam_model is not None:
        return
    import torch
    from transformers import SamModel, SamProcessor
    logger.info("[SAM] Loading SAM model (facebook/sam-vit-large)...")
    _sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-large")
    _sam_model = SamModel.from_pretrained("facebook/sam-vit-large")
    if torch.cuda.is_available():
        _sam_model = _sam_model.to("cuda")
    logger.info("[SAM] Model ready.")


def _unload_sam():
    global _sam_model, _sam_processor
    if _sam_model is None:
        return
    import torch, gc
    del _sam_model, _sam_processor
    _sam_model = None
    _sam_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[SAM] Model unloaded.")


@router.post("/mask/click")
async def click_mask(
    image: UploadFile = File(...),
    x: float = Form(...),
    y: float = Form(...),
):
    """SAM click-to-select mask: returns 3 masks (small/medium/large) with scores."""
    import asyncio
    try:
        img_bytes = await image.read()
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        def _run_sam(pil_img, click_x, click_y):
            import torch
            _load_sam()
            w, h = pil_img.size
            px = int(click_x * w)
            py = int(click_y * h)
            input_points = [[[px, py]]]

            inputs = _sam_processor(pil_img, input_points=input_points, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = _sam_model(**inputs)

            masks = _sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
            )

            scores = outputs.iou_scores.cpu()[0, 0]  # [3] scores
            all_masks = masks[0][0]  # [3, H, W]

            # Sort by mask area (smallest first) for intuitive S/M/L toggle
            areas = [all_masks[i].sum().item() for i in range(all_masks.shape[0])]
            sorted_indices = sorted(range(len(areas)), key=lambda i: areas[i])

            result_masks = []
            result_scores = []
            labels = ["S", "M", "L"]
            for idx, si in enumerate(sorted_indices):
                mask_np = all_masks[si].numpy().astype(np.uint8) * 255
                result_masks.append(_pil_to_base64(mask_np))
                result_scores.append(round(scores[si].item(), 3))

            # Keep SAM loaded between clicks — unloaded when diffusion pipeline starts.
            return result_masks, result_scores, labels

        masks_b64, scores, labels = await asyncio.to_thread(_run_sam, pil, x, y)
        return {
            "masks": masks_b64,
            "scores": scores,
            "labels": labels,
            "selected": 0,  # Default: smallest mask
            "status": "success",
        }
    except Exception as e:
        logger.exception("[SAM] click_mask failed")
        _unload_sam()
        return {"error": str(e), "status": "error"}

