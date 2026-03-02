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
    PUBLIC_DEMO, PUBLIC_MAX_STEPS, WEIGHTS_DIR, BASE_DIR, COMPILE_UNET,
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

    use_cpu_offload = force_cpu_offload or CPU_OFFLOAD

    if use_cpu_offload:
        # CPU offload: keeps model on CPU, moves layers to GPU on-demand
        # IMPORTANT: do NOT call p.to("cuda") before this — they're incompatible
        p.enable_model_cpu_offload()
        print(f"  [{pipeline_name}] ✓ CPU offload enabled (saves ~4-6GB VRAM)")
    else:
        p.to("cuda")

    # VAE tiling: DO NOT enable with CPU offload — the combination forces the VAE
    # to move CPU↔GPU once per tile (4-16 trips at 1024px), causing 10+ minute
    # hangs after the last diffusion step. Only enable for direct-CUDA pipelines
    # where the image is too large to decode in one pass (>1536px).
    if not use_cpu_offload:
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

    # Flash SDP: global PyTorch backend setting — safe regardless of CPU offload mode.
    # Enable unconditionally so attention ops use Flash Attention when on GPU.
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    except Exception:
        pass

    # xformers: try for all pipelines (compatible with model_cpu_offload in practice —
    # xformers hooks module.forward, accelerate hooks the module itself; no conflict).
    try:
        p.enable_xformers_memory_efficient_attention()
        print(f"  [{pipeline_name}] ✓ xformers enabled")
    except Exception:
        print(f"  [{pipeline_name}] ✓ Flash SDP enabled (xformers not available)")

    return p


def _unload_aux_models():
    """Unload SAM and SegFormer from VRAM before loading diffusion models."""
    try:
        from ..routers.mask import _unload_sam
        _unload_sam()
    except Exception:
        pass
    try:
        from segformer_masks import _unload_segformer
        _unload_segformer()
    except Exception:
        pass


def get_inpaint_pipe():
    """Load or return the SDXL inpaint pipeline."""
    global pipe, PIPE, txt2img_pipe
    if pipe is not None:
        return pipe

    # Free SAM / SegFormer VRAM before loading SDXL
    _unload_aux_models()

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

        # Optional: torch.compile for 20-40% per-step speedup (COMPILE_UNET=1)
        # First-run compilation takes 60-120s; subsequent calls are fast.
        if COMPILE_UNET and DEVICE == "cuda" and hasattr(torch, "compile"):
            try:
                p.unet = torch.compile(p.unet, mode="reduce-overhead")
                print(f"[PIPE] ✓ torch.compile applied to SDXL UNet")
            except Exception as e:
                print(f"[PIPE] torch.compile skipped: {e}")

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

    # Free SAM / SegFormer VRAM before loading FLUX
    _unload_aux_models()

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

        # VAE tiling disabled: CPU offload + tiling causes per-tile CPU↔GPU moves
        # which can add minutes of decode time. FLUX at ≤1024px fits in VRAM without tiling.
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
