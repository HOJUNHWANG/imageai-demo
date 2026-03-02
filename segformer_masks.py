"""
SegFormer-based clothing segmentation for precise auto masks.
Uses mattmdjaga/segformer_b2_clothes for 18-class semantic segmentation.
Loaded on-demand and unloaded after use to conserve VRAM (~400MB).

Classes (ATR dataset):
0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes,
5=Skirt, 6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe,
11=Face, 12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm,
16=Bag, 17=Scarf
"""
import gc
import numpy as np
from PIL import Image

_segformer_model = None
_segformer_processor = None


def _load_segformer():
    global _segformer_model, _segformer_processor
    if _segformer_model is not None:
        return
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    print("[SegFormer] Loading segformer_b2_clothes...")
    _segformer_processor = SegformerImageProcessor.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    _segformer_model = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )
    if torch.cuda.is_available():
        _segformer_model = _segformer_model.to("cuda")
    _segformer_model.eval()
    print("[SegFormer] Model ready.")


def _unload_segformer():
    global _segformer_model, _segformer_processor
    if _segformer_model is None:
        return
    import torch
    del _segformer_model, _segformer_processor
    _segformer_model = None
    _segformer_processor = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[SegFormer] Model unloaded.")


def _get_segmentation_map(pil: Image.Image) -> np.ndarray:
    """Run SegFormer and return full segmentation map (H, W) with class IDs."""
    import torch
    _load_segformer()
    inputs = _segformer_processor(images=pil, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _segformer_model(**inputs)

    logits = outputs.logits  # [1, num_classes, h, w]
    upsampled = torch.nn.functional.interpolate(
        logits, size=pil.size[::-1], mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)
    # Keep SegFormer loaded between requests — unloaded when diffusion pipeline starts.
    return seg_map


# ─── Class mappings ───
UPPER_CLASSES = {4}          # Upper-clothes
LOWER_CLASSES = {5, 6}       # Skirt, Pants
DRESS_CLASSES = {7}           # Full-body dress
HAIR_CLASSES = {2}            # Hair
FACE_CLASSES = {11}           # Face
ACCESSORIES = {1, 3, 8, 16, 17}  # Hat, Sunglasses, Belt, Bag, Scarf
PERSON_CLASSES = {2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 17}  # All body parts + clothes


def _classes_to_mask(seg_map: np.ndarray, class_ids: set) -> np.ndarray:
    """Convert segmentation map to binary mask for given class IDs."""
    mask = np.zeros(seg_map.shape, dtype=np.uint8)
    for cid in class_ids:
        mask[seg_map == cid] = 255
    return mask


def _morph_clean(mask: np.ndarray, k: int = 5) -> np.ndarray:
    """Morphological close to clean up small holes."""
    try:
        import cv2
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask
    except ImportError:
        from PIL import ImageFilter
        img = Image.fromarray(mask)
        img = img.filter(ImageFilter.MaxFilter(size=k))
        img = img.filter(ImageFilter.MinFilter(size=k))
        return np.array(img, dtype=np.uint8)


# ─── Public API ───

def segformer_top_mask(pil: Image.Image) -> np.ndarray:
    """Precise upper-body clothes mask using SegFormer."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, UPPER_CLASSES)
    return _morph_clean(mask)


def segformer_bottom_mask(pil: Image.Image) -> np.ndarray:
    """Precise lower-body clothes mask (pants/skirt) using SegFormer."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, LOWER_CLASSES)
    return _morph_clean(mask)


def segformer_dress_mask(pil: Image.Image) -> np.ndarray:
    """Full-body dress mask using SegFormer."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, DRESS_CLASSES | UPPER_CLASSES | LOWER_CLASSES)
    return _morph_clean(mask)


def segformer_hair_mask(pil: Image.Image) -> np.ndarray:
    """Precise hair mask using SegFormer."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, HAIR_CLASSES)
    return _morph_clean(mask, k=3)


def segformer_face_mask(pil: Image.Image) -> np.ndarray:
    """Face mask using SegFormer."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, FACE_CLASSES)
    return _morph_clean(mask, k=3)


def segformer_accessories_mask(pil: Image.Image) -> np.ndarray:
    """Accessories mask (hat, sunglasses, belt, bag, scarf)."""
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, ACCESSORIES)
    return _morph_clean(mask, k=3)


def segformer_face_hair_mask(pil: Image.Image) -> np.ndarray:
    """Combined face + hair mask from a single SegFormer inference.
    Used by protect_face to avoid running SegFormer twice for the same image.
    """
    seg_map = _get_segmentation_map(pil)
    mask = _classes_to_mask(seg_map, FACE_CLASSES | HAIR_CLASSES)
    return _morph_clean(mask, k=3)
