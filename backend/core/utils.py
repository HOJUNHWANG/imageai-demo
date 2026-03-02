"""
Shared utility functions for the ImageAI Studio backend.
"""
import io
import base64
import numpy as np
from PIL import Image


def pil_to_base64(img: "Image.Image | np.ndarray", format: str = "JPEG", quality: int = 90) -> str:
    """Convert PIL image or numpy array to base64-encoded string."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buffer = io.BytesIO()
    if format == "JPEG":
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
    else:
        img.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode()
