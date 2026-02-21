"""
AI Pipeline management: loading, switching, and unloading models.
Optimized for RTX 3080 Ti (12GB VRAM) — performance/quality balance.
"""
import os
import gc
import time
import torch
import numpy as np
from PIL import Image, ImageFilter
from typing import Optional, Generator

from diffusers import (
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline,
    FluxPipeline,
)

from .config import (
    DEVICE, MOCK_INPAINT, JUGGERNAUT_INPAINT, DEFAULT_MODEL,
    CONTROLNET_DEPTH, CONTROLNET_OPENPOSE, CONTROLNET_CANNY,
    LOW_VRAM, CPU_OFFLOAD, AUTO_UNLOAD_AUX, AUTO_HARD_CLEAR_THRESHOLD,
    PUBLIC_DEMO, PUBLIC_MAX_STEPS, WEIGHTS_DIR, BASE_DIR,
)

try:
    import cv2
except Exception:
    cv2 = None

# ─── Global state ───
pipe = None
PIPE = None
controlnet_pipes: dict = {}
img2img_pipe = None
txt2img_pipe = None
CURRENT_MODE = "edit"

# Working image state
STATE = {
    "working_pil": None,
    "working_np": None,
    "orig_pil": None,
    "manual_mask_u8": None,
    "auto_mask_u8": None,
    "active_mask_source": None,
    "selected_mask": None,
    "auto_mask_candidates": [],
}


def _aggressive_vram_cleanup():
    """Aggressive VRAM cleanup between model switches."""
    gc.collect()
    gc.collect()  # Double collect for cyclic references
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        # Reset peak stats
        torch.cuda.reset_peak_memory_stats()


def unload_aux_pipelines():
    """Unload optional pipelines (ControlNet + Refine) to recover VRAM."""
    global controlnet_pipes, img2img_pipe
    if isinstance(controlnet_pipes, dict):
        controlnet_pipes.clear()
    img2img_pipe = None
    _aggressive_vram_cleanup()


def hard_clear_vram():
    """Hard clear: unload all models from GPU."""
    global pipe, PIPE, controlnet_pipes, img2img_pipe, txt2img_pipe
    pipe = None
    PIPE = None
    txt2img_pipe = None
    if isinstance(controlnet_pipes, dict):
        controlnet_pipes.clear()
    else:
        controlnet_pipes = {}
    img2img_pipe = None
    _aggressive_vram_cleanup()
    return {"message": "[VRAM][HARD] All pipelines unloaded."}


def _apply_optimizations(p, pipeline_name="pipeline", force_cpu_offload=False):
    """Apply GPU optimizations. Use force_cpu_offload=True for large models (FLUX)."""
    if DEVICE != "cuda":
        p.to("cpu")
        return p

    if force_cpu_offload or CPU_OFFLOAD:
        # CPU offload: keeps model on CPU, moves layers to GPU on-demand
        # IMPORTANT: do NOT call p.to("cuda") before this — they're incompatible
        p.enable_model_cpu_offload()
        print(f"  [{pipeline_name}] ✓ CPU offload enabled (saves ~4-6GB VRAM)")
    else:
        p.to("cuda")

    # VAE optimizations (critical for 12GB cards)
    try:
        p.enable_vae_tiling()
        print(f"  [{pipeline_name}] ✓ VAE tiling enabled")
    except Exception:
        pass

    try:
        p.enable_vae_slicing()
        print(f"  [{pipeline_name}] ✓ VAE slicing enabled")
    except Exception:
        pass

    # Attention slicing (saves VRAM at slight speed cost)
    try:
        p.enable_attention_slicing("auto")
        print(f"  [{pipeline_name}] ✓ Attention slicing enabled")
    except Exception:
        pass

    # xformers (fastest attention, only if not using CPU offload)
    if not force_cpu_offload:
        try:
            p.enable_xformers_memory_efficient_attention()
            print(f"  [{pipeline_name}] ✓ xformers enabled")
        except Exception:
            pass

    return p


def get_inpaint_pipe():
    """Load or return the SDXL inpaint pipeline."""
    global pipe, PIPE, txt2img_pipe
    if pipe is not None:
        return pipe

    # Free txt2img if loaded
    if txt2img_pipe is not None:
        print("[PIPE] Unloading FLUX to free VRAM for SDXL...")
        txt2img_pipe = None
        _aggressive_vram_cleanup()

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL

    print(f"[PIPE] Loading Inpaint from: {model_path}")
    t0 = time.time()

    try:
        if os.path.exists(model_path):
            p = StableDiffusionXLInpaintPipeline.from_single_file(
                model_path, torch_dtype=dtype, use_safetensors=True, safety_checker=None,
            )
        else:
            p = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_path, torch_dtype=dtype, safety_checker=None,
            )

        p = _apply_optimizations(p, "SDXL")
        pipe = p
        PIPE = p

        print(f"[PIPE] SDXL ready in {time.time()-t0:.1f}s")
        return pipe
    except Exception as e:
        print(f"[PIPE] SDXL Load Failed: {e}")
        return None


def get_txt2img_pipe():
    """Load or return the FLUX text-to-image pipeline.
    Uses 4-bit quantization (NF4) to fit in 12GB VRAM.
    """
    global pipe, PIPE, txt2img_pipe
    if txt2img_pipe is not None:
        return txt2img_pipe

    # Unload SDXL to free all VRAM
    if pipe is not None:
        print("[PIPE] Unloading SDXL to free VRAM for FLUX...")
        pipe = None
        PIPE = None
        _aggressive_vram_cleanup()

    t0 = time.time()
    model_id = "black-forest-labs/FLUX.1-schnell"

    try:
        # Try 4-bit quantization first (reduces ~23GB → ~6GB)
        try:
            from diffusers import BitsAndBytesConfig as DiffusersBnBConfig
            quant_config = DiffusersBnBConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            print("[FLUX] Loading with 4-bit quantization (NF4)...")
            from diffusers import FluxTransformer2DModel
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16,
            )
            txt2img_pipe = FluxPipeline.from_pretrained(
                model_id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
            )
            txt2img_pipe.enable_model_cpu_offload()
            print(f"  [FLUX] ✓ 4-bit quantization + CPU offload (~6GB VRAM)")
        except Exception as quant_err:
            print(f"  [FLUX] ⚠ 4-bit quantization failed: {quant_err}")
            print("[FLUX] Falling back to bfloat16 with sequential CPU offload...")
            txt2img_pipe = FluxPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16,
            )
            # sequential offload: moves individual layers (not modules), much less VRAM
            txt2img_pipe.enable_sequential_cpu_offload()
            print(f"  [FLUX] ✓ Sequential CPU offload (slowest, but fits any GPU)")

        # Always apply VAE optimizations
        try:
            txt2img_pipe.enable_vae_tiling()
        except Exception:
            pass
        try:
            txt2img_pipe.enable_vae_slicing()
        except Exception:
            pass
        try:
            txt2img_pipe.enable_attention_slicing("auto")
        except Exception:
            pass

        print(f"[FLUX] Pipeline ready in {time.time()-t0:.1f}s")
        return txt2img_pipe
    except Exception as e:
        print(f"[FLUX] Load failed: {e}")
        return None


def switch_mode(target_mode: str) -> str:
    global CURRENT_MODE
    print(f"[MODE] Switching to {target_mode}...")
    if target_mode == "generate":
        get_txt2img_pipe()
        CURRENT_MODE = "generate"
        return "Switched to Text-to-Image (FLUX)"
    else:
        get_inpaint_pipe()
        CURRENT_MODE = "edit"
        return "Switched to Inpainting (SDXL)"
