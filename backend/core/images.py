"""Image decoding, model sizing and exact masked compositing."""
from __future__ import annotations

import base64
import io
import math

from PIL import Image, ImageFilter, ImageOps

from .config import MAX_INPUT_PIXELS, MAX_LONG_SIDE, MAX_PIXELS


def _validate_input_size(image: Image.Image) -> None:
    width, height = image.size
    pixels = width * height
    if width <= 0 or height <= 0 or pixels > MAX_INPUT_PIXELS:
        raise ValueError(
            f"Image dimensions exceed the {MAX_INPUT_PIXELS:,}-pixel input limit"
        )


def decode_image(data: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(data))
    _validate_input_size(image)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def decode_mask(data: bytes, size: tuple[int, int]) -> Image.Image:
    raw = Image.open(io.BytesIO(data))
    _validate_input_size(raw)
    if "A" in raw.getbands():
        alpha = raw.getchannel("A")
        # Canvas masks use transparency outside the painted area.
        if alpha.getextrema() != (255, 255):
            raw = alpha
        else:
            raw = raw.convert("L")
    else:
        raw = raw.convert("L")
    return raw.resize(size, Image.Resampling.NEAREST)


def model_size(
    size: tuple[int, int],
    requested_long_side: int,
    requested_max_pixels: int | None = None,
) -> tuple[int, int]:
    width, height = size
    long_side = min(max(512, requested_long_side), MAX_LONG_SIDE)
    max_pixels = min(requested_max_pixels or MAX_PIXELS, MAX_PIXELS)
    scale = min(long_side / max(width, height), math.sqrt(max_pixels / (width * height)))
    out_w = max(64, int(width * scale) // 32 * 32)
    out_h = max(64, int(height * scale) // 32 * 32)
    return out_w, out_h


def prepare_for_model(
    image: Image.Image,
    requested_long_side: int,
    requested_max_pixels: int | None = None,
) -> Image.Image:
    size = model_size(image.size, requested_long_side, requested_max_pixels)
    return image.resize(size, Image.Resampling.LANCZOS)


def composite_at_original_resolution(
    original: Image.Image,
    edited: Image.Image,
    mask: Image.Image | None,
    feather: int,
) -> Image.Image:
    edited = edited.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    if mask is None:
        return edited
    mask = mask.convert("L").resize(original.size, Image.Resampling.NEAREST)
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=min(feather, 64)))
    return Image.composite(edited, original.convert("RGB"), mask)


def encode_png(image: Image.Image) -> str:
    output = io.BytesIO()
    # Pillow's optimize pass is CPU-heavy and blocks delivery after GPU inference.
    image.save(output, format="PNG", compress_level=4)
    return base64.b64encode(output.getvalue()).decode("ascii")
