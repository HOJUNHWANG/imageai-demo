"""
VRAM and memory management utilities.
Extracted from app.py VRAM helpers.
"""
import gc
import torch
import psutil
from .config import DEVICE


def _gb(x: int) -> float:
    return float(x) / (1024 ** 3)


def get_vram_info(fast: bool = False) -> dict:
    """Return VRAM info as a structured dict.

    Args:
        fast: If True, skip synchronize() to avoid blocking the GPU stream
              during active inference. Suitable for polling endpoints.
    """
    if DEVICE != "cuda" or not torch.cuda.is_available():
        ram = psutil.virtual_memory()
        return {
            "gpu_name": "(CPU mode)",
            "allocated_gb": 0,
            "reserved_gb": 0,
            "max_allocated_gb": 0,
            "max_reserved_gb": 0,
            "total_gb": 0,
            "ram_used_gb": round(_gb(ram.used), 2),
            "ram_total_gb": round(_gb(ram.total), 2),
        }

    if not fast:
        torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    total = torch.cuda.get_device_properties(dev).total_memory

    return {
        "gpu_name": torch.cuda.get_device_name(dev),
        "allocated_gb": round(_gb(torch.cuda.memory_allocated(dev)), 2),
        "reserved_gb": round(_gb(torch.cuda.memory_reserved(dev)), 2),
        "max_allocated_gb": round(_gb(torch.cuda.max_memory_allocated(dev)), 2),
        "max_reserved_gb": round(_gb(torch.cuda.max_memory_reserved(dev)), 2),
        "total_gb": round(_gb(total), 2),
    }


def get_vram_text() -> str:
    """Formatted VRAM string for display."""
    info = get_vram_info()
    if info["gpu_name"] == "(CPU mode)":
        return (
            f"GPU: (CPU mode)\n"
            f"RAM Used: {info['ram_used_gb']:.2f} GB / {info['ram_total_gb']:.2f} GB\n"
        )
    return (
        f"GPU: {info['gpu_name']}\n"
        f"Allocated: {info['allocated_gb']:.2f} GB\n"
        f"Reserved:  {info['reserved_gb']:.2f} GB\n"
        f"MaxAlloc:  {info['max_allocated_gb']:.2f} GB\n"
        f"MaxResv:   {info['max_reserved_gb']:.2f} GB\n"
        f"Total:     {info['total_gb']:.2f} GB\n"
    )


def soft_clear_vram() -> dict:
    """Soft clear: cache only (models stay on GPU)."""
    msg = "[VRAM][SOFT] CPU mode (no CUDA)."
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            msg = "[VRAM][SOFT] empty_cache() + ipc_collect() done."
        except Exception as e:
            msg = f"[VRAM][SOFT] failed: {type(e).__name__}: {e}"
    return {"message": msg, "vram": get_vram_info()}
