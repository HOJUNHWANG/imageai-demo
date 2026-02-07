# venv311\Scripts\activate
# app.py
import os

# Memory behavior (can help reduce fragmentation/OOM on long-running sessions)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Debug only: enables more precise CUDA stack traces but slows down inference.
# Turn on explicitly when debugging CUDA issues.
if os.getenv("DEBUG_CUDA", "0") == "1":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

import time
import gc
import psutil  # RAM 사용량 확인용
import re
import numpy as np
from PIL import Image, ImageFilter
import torch
import traceback

# Speed boost on RTX 30/40 (minimal quality impact for inference)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from diffusers import (
    StableDiffusionXLInpaintPipeline,
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLImg2ImgPipeline
)
from diffusers.utils import load_image
from prompt_enricher import enrich_positive, enrich_negative

# opencv 있으면 더 좋고, 없으면 PIL로만 동작
try:
    import cv2
except Exception:
    cv2 = None

import gradio as gr

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "sans-serif"]
)

from sam_utils import SamMaskerManager
from mp_tasks_utils import MPTasksHelper
from mp_tasks_utils import (
    build_sleeve_mask_v5_tasks,
    build_top_mask_v5_tasks,
    build_pants_mask_v5_tasks,
    build_hair_mask_v5_tasks,
    build_background_mask_v5_tasks,
)

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODELS_DIR = os.path.join(BASE_DIR, "models", "stable-diffusion-xl")

JUGGERNAUT_INPAINT = os.path.join(MODELS_DIR, "juggernautXL_ragnarokBy.safetensors")
DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

CONTROLNET_DEPTH = os.path.join(BASE_DIR, "models", "ControlNet", "controlnet-depth-sdxl-1.0")
CONTROLNET_OPENPOSE = os.path.join(BASE_DIR, "models", "ControlNet", "controlnet-openpose-sdxl-1.0")

MOCK_INPAINT = os.getenv("MOCK_INPAINT", "0") == "1"

# -----------------------------------------------------------------------------
# Public demo safety / feature flags
# -----------------------------------------------------------------------------
# This repository is intended to be portfolio/public-demo friendly.
# Guardrails keep latency/VRAM stable on shared GPU demos.
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "1") == "1"

PUBLIC_MAX_LONG_SIDE = int(os.getenv("PUBLIC_MAX_LONG_SIDE", "896"))
PUBLIC_MAX_STEPS = int(os.getenv("PUBLIC_MAX_STEPS", "22"))
PUBLIC_MAX_QUEUE = int(os.getenv("PUBLIC_MAX_QUEUE", "10"))
PUBLIC_CONCURRENCY = int(os.getenv("PUBLIC_CONCURRENCY", "1"))

# VRAM/RAM stability toggles
# - LOW_VRAM enables VAE slicing/tiling + attention slicing (slower, more stable)
# - CPU_OFFLOAD enables diffusers model CPU offload (more stable, may be slower)
# - AUTO_UNLOAD_AUX unloads ControlNet/Refine pipelines after each run (frees VRAM)
LOW_VRAM = os.getenv("LOW_VRAM", "1" if PUBLIC_DEMO else "0") == "1"
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"
AUTO_UNLOAD_AUX = os.getenv("AUTO_UNLOAD_AUX", "1" if PUBLIC_DEMO else "0") == "1"

# Extra stability knobs
WARMUP = os.getenv("WARMUP", "0" if PUBLIC_DEMO else "1") == "1"
AUTO_HARD_CLEAR_THRESHOLD = float(os.getenv("AUTO_HARD_CLEAR_THRESHOLD", "0.92"))

# Public demo mode choices only

def get_edit_mode_choices():
    return ["Wear / Change Clothes", "General Edit"]

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
torch.set_float32_matmul_precision("high")

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}")

# -----------------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------------

STATE = {
    "working_pil": None,
    "working_np": None,
    "orig_pil": None,

    # Masks
    "manual_mask_u8": None,
    "auto_mask_u8": None,
    "active_mask_source": None,  # "manual" | "auto" | None

    "selected_mask": None,
    "auto_mask_candidates": []
}

# Global pipelines
pipe = None
PIPE = None  # 유지 (기존 코드 호환)
controlnet_pipes = {}
img2img_pipe = None

# -----------------------------------------------------------------------------
# VRAM / RAM helpers
# -----------------------------------------------------------------------------

def _gb(x: int) -> float:
    return float(x) / (1024 ** 3)

def get_vram_text() -> str:
    if DEVICE != "cuda" or (not torch.cuda.is_available()):
        ram = psutil.virtual_memory()
        return (
            f"GPU: (CPU mode)\n"
            f"RAM Used: {_gb(ram.used):.2f} GB / {_gb(ram.total):.2f} GB\n"
        )

    torch.cuda.synchronize()
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    alloc = torch.cuda.memory_allocated(dev)
    resv = torch.cuda.memory_reserved(dev)
    max_alloc = torch.cuda.max_memory_allocated(dev)
    max_resv = torch.cuda.max_memory_reserved(dev)
    total = torch.cuda.get_device_properties(dev).total_memory

    return (
        f"GPU: {name}\n"
        f"Allocated: {_gb(alloc):.2f} GB\n"
        f"Reserved:  {_gb(resv):.2f} GB\n"
        f"MaxAlloc:  {_gb(max_alloc):.2f} GB\n"
        f"MaxResv:   {_gb(max_resv):.2f} GB\n"
        f"Total:     {_gb(total):.2f} GB\n"
    )

def soft_clear_vram():
    """
    🧹 Soft Clear: 캐시만 비움 (모델은 GPU에 유지 → 2번째 속도 유지)
    """
    msg = "[VRAM][SOFT] CPU mode (no CUDA)."
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # ipc_collect는 드물게 도움됨(특히 메모리 파편화/보류된 블록)
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
            msg = "[VRAM][SOFT] empty_cache() + ipc_collect() done."
        except Exception as e:
            msg = f"[VRAM][SOFT] failed: {type(e).__name__}: {e}"
    return msg, get_vram_text()

def unload_aux_pipelines():
    """Unload optional pipelines (ControlNet + Refine) to recover VRAM."""
    global controlnet_pipes, img2img_pipe

    # ControlNet pipelines
    if isinstance(controlnet_pipes, dict) and controlnet_pipes:
        for k, p in list(controlnet_pipes.items()):
            try:
                if p is not None and hasattr(p, "to"):
                    p.to("cpu")
            except Exception:
                pass
        controlnet_pipes.clear()

    # Refine pipeline
    if img2img_pipe is not None:
        try:
            if hasattr(img2img_pipe, "to"):
                img2img_pipe.to("cpu")
        except Exception:
            pass
        img2img_pipe = None

    gc.collect()
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            pass


def hard_clear_vram():
    """
    🔥 Hard Clear: 모델/ControlNet/Refine 파이프를 GPU에서 내리고 객체 제거
    - OOM 복구 확실
    - 다음 생성은 모델 재로딩 때문에 느려짐
    """
    global pipe, PIPE, controlnet_pipes, img2img_pipe

    if DEVICE != "cuda" or (not torch.cuda.is_available()):
        # CPU 모드면 그냥 객체만 정리
        pipe = None
        PIPE = None
        controlnet_pipes.clear()
        img2img_pipe = None
        gc.collect()
        return "[VRAM][HARD] CPU mode cleanup done.", get_vram_text()

    try:
        torch.cuda.synchronize()

        # 1) 메인 파이프라인 unload
        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
        pipe = None
        PIPE = None

        # 2) ControlNet 파이프들 unload
        if isinstance(controlnet_pipes, dict) and controlnet_pipes:
            for k, p in list(controlnet_pipes.items()):
                try:
                    p.to("cpu")
                except Exception:
                    pass
            controlnet_pipes.clear()

        # 3) Refine(img2img) unload
        if img2img_pipe is not None:
            try:
                img2img_pipe.to("cpu")
            except Exception:
                pass
        img2img_pipe = None

        # 4) 파이썬/토치 정리
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()

        msg = "[VRAM][HARD] Unloaded pipelines + cleared CUDA cache. (Next run will reload models)"
        return msg, get_vram_text()

    except Exception as e:
        return f"[VRAM][HARD] failed: {type(e).__name__}: {e}", get_vram_text()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def normalize_space(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def comma_join_unique(items):
    out = []
    seen = set()
    for x in items:
        x = normalize_space(x).strip(",")
        if not x:
            continue
        key = x.lower()
        if key not in seen:
            out.append(x)
            seen.add(key)
    return ", ".join(out)

def build_default_negative(mode: str) -> str:
    base = [
        "low quality", "blurry", "jpeg artifacts", "bad anatomy", "deformed",
        "extra arms", "extra hands", "extra fingers", "missing fingers",
        "plastic skin", "over-smoothed skin", "uncanny", "text", "watermark", "logo"
    ]

    # Public/portfolio-friendly defaults
    if mode == "Wear / Change Clothes":
        return comma_join_unique(base)
    return comma_join_unique(base)

def resize_to_long_side(pil: Image.Image, long_side: int) -> Image.Image:
    w, h = pil.size
    if max(w, h) == long_side:
        return pil
    scale = long_side / float(max(w, h))
    nw = int(round(w * scale))
    nh = int(round(h * scale))
    return pil.resize((nw, nh), Image.LANCZOS)

def to_rgb_np(pil: Image.Image) -> np.ndarray:
    return np.array(pil.convert("RGB"), dtype=np.uint8)

def overlay_mask(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    vis = image_rgb.copy()
    red = np.zeros_like(vis)
    red[..., 0] = 255
    alpha = 0.45
    m = (mask_u8 > 0)
    vis[m] = (vis[m] * (1 - alpha) + red[m] * alpha).astype(np.uint8)
    return vis

def postprocess_mask(mask_u8: np.ndarray, expand_px: int, blur_px: int) -> np.ndarray:
    m = (mask_u8 > 0).astype(np.uint8) * 255

    if expand_px > 0:
        k = expand_px * 2 + 1
        m = cv2.dilate(m, np.ones((k, k), np.uint8), iterations=1) if cv2 is not None else m

    if blur_px > 0:
        k = blur_px * 2 + 1
        m = cv2.GaussianBlur(m, (k, k), 0) if cv2 is not None else m

    return np.clip(m, 0, 255).astype(np.uint8)

def _get_tokenizer():
    # Tokenizer is available only after the pipeline is loaded.
    if pipe is not None and hasattr(pipe, "tokenizer"):
        return pipe.tokenizer
    return None


def _split_sdxl_prompt_overflow(text: str, tokenizer, max_len: int = 77):
    """Return (prompt, prompt_2, token_count, token_count_2, warnings).

    Strategy: split by commas into two prompts when token count exceeds max_len.
    If still too long, truncate safely.
    """
    warnings = []
    text = normalize_space(text)
    if tokenizer is None:
        return text, None, None, None, ["Tokenizer unavailable (model not loaded). Prompt overflow check skipped."]

    def count_tokens(t: str) -> int:
        ids = tokenizer(t, add_special_tokens=True, return_tensors=None)["input_ids"]
        return len(ids)

    def truncate_to(t: str, limit: int) -> str:
        out = tokenizer(t, add_special_tokens=True, truncation=True, max_length=limit, return_tensors=None)
        return tokenizer.decode(out["input_ids"], skip_special_tokens=True)

    t0 = text
    n0 = count_tokens(t0)
    if n0 <= max_len:
        return t0, None, n0, 0, []

    parts = [p.strip() for p in t0.split(",") if p.strip()]
    if len(parts) <= 1:
        # no safe split point; truncate
        warnings.append(f"Prompt exceeds {max_len} tokens ({n0}). Truncated.")
        t_tr = truncate_to(t0, max_len)
        return t_tr, None, count_tokens(t_tr), 0, warnings

    # greedy split: build prompt1 until near limit
    p1 = []
    p2 = []
    for p in parts:
        cand = ", ".join(p1 + [p])
        if count_tokens(cand) <= max_len:
            p1.append(p)
        else:
            p2.append(p)

    s1 = ", ".join(p1).strip()
    s2 = ", ".join(p2).strip() if p2 else None

    n1 = count_tokens(s1) if s1 else 0
    n2 = count_tokens(s2) if s2 else 0

    if s2 is None:
        warnings.append(f"Prompt exceeds {max_len} tokens ({n0}). Truncated.")
        s1 = truncate_to(t0, max_len)
        return s1, None, count_tokens(s1), 0, warnings

    if n2 > max_len:
        warnings.append(f"prompt_2 exceeds {max_len} tokens ({n2}). Truncated.")
        s2 = truncate_to(s2, max_len)
        n2 = count_tokens(s2)

    warnings.append(f"Prompt overflow: split into prompt + prompt_2 (max {max_len} tokens each).")
    return s1, s2, n1, n2, warnings


def build_final_prompts(prompt: str, negative: str, auto_enrich: bool, edit_mode: str):
    tok = _get_tokenizer()

    pos = (prompt or "").strip()
    neg = (negative or "").strip()

    if auto_enrich:
        try:
            pos, _ = enrich_positive(pos)
        except Exception as e:
            print(f"[WARN] Positive enrich failed: {e}")
        try:
            neg = enrich_negative(neg)
        except Exception as e:
            print(f"[WARN] Negative enrich failed: {e}")

    neg = comma_join_unique([neg, build_default_negative(edit_mode)])

    # SDXL 77-token handling
    pos1, pos2, pos_n1, pos_n2, pos_warn = _split_sdxl_prompt_overflow(pos, tok, 77)
    neg1, neg2, neg_n1, neg_n2, neg_warn = _split_sdxl_prompt_overflow(neg, tok, 77)

    warnings = pos_warn + neg_warn

    return {
        "prompt": pos1,
        "prompt_2": pos2,
        "negative": neg1,
        "negative_2": neg2,
        "tok_pos": pos_n1,
        "tok_pos2": pos_n2,
        "tok_neg": neg_n1,
        "tok_neg2": neg_n2,
        "warnings": warnings,
    }


def preview_enriched_prompt(prompt: str, negative: str, auto_enrich: bool, edit_mode: str):
    if not (prompt or "").strip():
        return "Positive prompt is required!"

    fin = build_final_prompts(prompt, negative, auto_enrich, edit_mode)

    warn = "\n".join([f"- {w}" for w in fin["warnings"]]) if fin["warnings"] else "- (none)"

    preview_md = f"""### Prompt Check

**Positive**

```
{fin['prompt']}
```

**Positive 2 (SDXL prompt_2)**

```
{fin['prompt_2'] or ''}
```

**Negative**

```
{fin['negative']}
```

**Negative 2 (SDXL negative_prompt_2)**

```
{fin['negative_2'] or ''}
```

**Token counts (approx)**
- pos: {fin['tok_pos']}
- pos2: {fin['tok_pos2']}
- neg: {fin['tok_neg']}
- neg2: {fin['tok_neg2']}

**Warnings**
{warn}
"""
    return preview_md

def parse_prompt_simple(prompt: str) -> dict:
    p = (prompt or "").lower()

    sleeve_kw = ["sleeve", "sleeveless", "tank", "crop top", "short sleeve", "long sleeve"]
    top_kw = ["shirt", "t-shirt", "top", "blouse", "jacket", "hoodie", "sweater"]

    if any(k in p for k in sleeve_kw):
        target = "sleeve"
    elif any(k in p for k in top_kw):
        target = "top"
    else:
        target = "top"

    color = None
    for c in ["black", "white", "red", "blue", "green", "gray", "brown", "beige"]:
        if c in p:
            color = c
            break

    garment = None
    for g in ["tank top", "t-shirt", "shirt", "blouse", "jacket", "hoodie", "sweater"]:
        if g in p:
            garment = g
            break

    return {"target": target, "color": color or "n/a", "garment": garment or "n/a"}

# -----------------------------------------------------------------------------
# Model loaders
# -----------------------------------------------------------------------------

def load_pipe():
    global pipe, PIPE
    if MOCK_INPAINT:
        print("[PIPE] MOCK_INPAINT mode - no real model loaded")
        pipe = None
        PIPE = None
        return None

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL

    print(f"[PIPE] Loading checkpoint from: {model_path}")
    print(f"[PIPE] Target device: {DEVICE}")
    print(f"[PIPE] Using dtype: {dtype}")

    try:
        if os.path.exists(model_path):
            # Local single-file checkpoint
            p = StableDiffusionXLInpaintPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
                variant="fp16" if "safetensors" in model_path.lower() and DEVICE == "cuda" else None,
                use_safetensors=True,
                safety_checker=None,
            )
        else:
            # HF repo id
            p = StableDiffusionXLInpaintPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                safety_checker=None,
            )

        if DEVICE == "cuda":
            p.to("cuda")

            # Offload vs full-GPU
            if CPU_OFFLOAD and hasattr(p, "enable_model_cpu_offload"):
                p.enable_model_cpu_offload()
                print("[GPU] CPU_OFFLOAD enabled (more stable, may be slower)")
            else:
                if hasattr(p, "disable_model_cpu_offload"):
                    p.disable_model_cpu_offload()
                print("[GPU] Full GPU mode (cpu_offload disabled)")

            # Low VRAM helpers
            if LOW_VRAM:
                try:
                    p.enable_attention_slicing("auto")
                except Exception:
                    pass
                try:
                    p.enable_vae_slicing()
                except Exception:
                    pass
                try:
                    p.enable_vae_tiling()
                except Exception:
                    pass
                print("[OPT] LOW_VRAM enabled (attention/vae slicing/tiling)")

            # Optional: xformers
            try:
                p.enable_xformers_memory_efficient_attention()
                print("[OPT] xformers enabled - faster attention")
            except Exception:
                print("[OPT] xformers not available (fallback)")
        else:
            p.to("cpu")
            print("[CPU] Running on CPU - generation will be slow")

        # Warm-up (optional)
        if WARMUP:
            print("[PIPE] Running warm-up dummy inference...")
            try:
                dummy_image = Image.new("RGB", (512, 512), color="white")
                dummy_mask = Image.new("L", (512, 512), color=0)
                _ = p(
                    prompt="a photo of a cat",
                    image=dummy_image,
                    mask_image=dummy_mask,
                    num_inference_steps=1,
                    strength=0.01,
                    guidance_scale=1.0,
                ).images[0]
                print("[PIPE] Warm-up success! GPU/VRAM ready")
            except Exception as warm_up_e:
                print(f"[PIPE] Warm-up failed (non-fatal): {str(warm_up_e)}")
        else:
            print("[PIPE] Warm-up disabled (WARMUP=0)")

        pipe = p
        PIPE = p
        print("[PIPE] Loaded successfully!")
        print(f"[PIPE] Pipeline class: {type(pipe).__name__}")
        print(f"[PIPE] Scheduler: {type(pipe.scheduler).__name__}")
        print(f"[PIPE] Device: {pipe.device}")
        print(f"[PIPE] UNet dtype: {next(pipe.unet.parameters()).dtype}")

        return pipe

    except Exception as e:
        print(f"[PIPE] Load failed: {str(e)}")
        print("[PIPE] Check model path, safetensors file, or diffusers version")
        pipe = None
        PIPE = None
        return None

PIPE = load_pipe()

def _boot_warnings():
    # MediaPipe Tasks model (optional)
    mp_model = os.path.join(WEIGHTS_DIR, "selfie_multiclass_256x256.tflite")
    if not os.path.exists(mp_model):
        print(f"[BOOT][WARN] MediaPipe model missing: {mp_model}")
        print("[BOOT][WARN] Auto-mask (MediaPipe) will be unavailable until you add it.")

    # SAM weights (optional)
    sam_b = os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth")
    sam_h = os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth")
    if not os.path.exists(sam_b) and not os.path.exists(sam_h):
        print(f"[BOOT][WARN] SAM weights not found under: {WEIGHTS_DIR}")
        print("[BOOT][WARN] Manual click-to-mask (SAM) will fail until weights are added.")

_boot_warnings()

mp_helper = None
try:
    mp_helper = MPTasksHelper(weights_dir=WEIGHTS_DIR)
except Exception as e:
    print(f"[BOOT][WARN] MediaPipe init failed: {e}")

sam_manager = SamMaskerManager(weights_dir=WEIGHTS_DIR, device=DEVICE)

# -----------------------------------------------------------------------------
# Gradio callbacks
# -----------------------------------------------------------------------------

def on_upload(img: Image.Image, working_long_side: int):
    if img is None:
        return None, None, "Upload an image first.", "Error: No image uploaded."

    # Keep memory stable: store only the resized working image by default.
    # (The original full-resolution image can be very large and can push RAM over the limit.)
    orig = img.convert("RGB")
    target_long = int(working_long_side)
    if PUBLIC_DEMO:
        target_long = min(target_long, PUBLIC_MAX_LONG_SIDE)
    working = resize_to_long_side(orig, target_long)

    STATE["orig_pil"] = None
    STATE["working_pil"] = working
    STATE["working_np"] = to_rgb_np(working)
    STATE["manual_mask_u8"] = None
    STATE["auto_mask_u8"] = None
    STATE["active_mask_source"] = None
    STATE["selected_mask"] = None
    STATE["auto_mask_candidates"] = []

    # release refs
    del orig
    gc.collect()

    status_msg = "Image loaded. Choose Auto Mask or click for Manual Mask."
    return working, None, status_msg, status_msg

def on_manual_click(evt: gr.SelectData, sam_model_type: str):
    if STATE["working_np"] is None:
        return None, None, "Upload an image first.", "Error: No working image."

    x, y = evt.index[0], evt.index[1]

    try:
        masker = sam_manager.get(sam_model_type)
        mask_u8 = masker.predict_from_click(STATE["working_np"], x, y)
    except Exception as e:
        # Friendly UI error instead of crashing the Gradio worker
        msg = (
            "Manual mask unavailable.\n"
            "- Install: pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "- Download weights into ./weights/: sam_vit_b_01ec64.pth (or sam_vit_h_4b8939.pth)\n"
            f"- Error: {type(e).__name__}: {e}"
        )
        return None, None, msg, msg

    STATE["manual_mask_u8"] = mask_u8
    STATE["active_mask_source"] = "manual"
    STATE["selected_mask"] = mask_u8

    vis = overlay_mask(STATE["working_np"], mask_u8)
    mask_preview = Image.fromarray(mask_u8)

    status_msg = f"Manual mask built (SAM {sam_model_type})."
    return Image.fromarray(vis), mask_preview, status_msg, status_msg

def build_auto_candidates_v5(prompt: str, auto_enrich: bool, edit_mode: str, auto_target: str):
    t0 = time.time()

    user_prompt = normalize_space(prompt)
    auto_enrich_flag = bool(auto_enrich)

    if STATE.get("working_pil") is None or STATE.get("working_np") is None:
        return [], "Upload an image first.", ""

    try:
        positive_final = user_prompt
        if auto_enrich_flag:
            positive_final, _info = enrich_positive(user_prompt)
        positive_final = normalize_space(positive_final)
    except Exception as e:
        return [], str(e), ""

    # Target selection: UI dropdown wins (more predictable than prompt parsing)
    target = (auto_target or "top").strip().lower()
    if target == "auto":
        try:
            info = parse_prompt_simple(positive_final)
            target = info.get("target", "top")
        except Exception as e:
            return [], str(e), positive_final

    if mp_helper is None:
        return [], "mp_helper unavailable", positive_final

    try:
        if target == "person":
            c = mp_helper.person_mask(STATE["working_pil"], threshold=0.5)
        elif target == "sleeve":
            c = build_sleeve_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "top":
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "pants":
            c = build_pants_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "hair":
            c = build_hair_mask_v5_tasks(STATE["working_pil"], mp_helper)
        elif target == "background":
            c = build_background_mask_v5_tasks(STATE["working_pil"], mp_helper)
        else:
            c = build_top_mask_v5_tasks(STATE["working_pil"], mp_helper)

        if c.ndim == 3 and c.shape[-1] == 1:
            c = c[..., 0]

        if c.dtype != np.uint8:
            c = c.astype(np.uint8)

        c = np.where(c > 127, 255, 0).astype(np.uint8)

        STATE["auto_mask_candidates"] = [c]
        STATE["auto_mask_u8"] = c
        STATE["active_mask_source"] = "auto"
        STATE["selected_mask"] = c  # auto 선택

        dt = time.time() - t0
        return [Image.fromarray(c)], f"v5 OK target={target} time={dt:.2f}s", positive_final

    except Exception as e:
        return [], str(e), positive_final

def _get_active_mask() -> np.ndarray | None:
    src = STATE.get("active_mask_source")
    if src == "manual":
        return STATE.get("manual_mask_u8")
    if src == "auto":
        return STATE.get("auto_mask_u8")
    # fallback
    return STATE.get("manual_mask_u8") or STATE.get("auto_mask_u8")


def _mask_source_text() -> str:
    src = STATE.get("active_mask_source")
    if src == "manual":
        return "Active mask: **Manual (SAM click)**"
    if src == "auto":
        return "Active mask: **Auto (MediaPipe)**"
    return "Active mask: *(none)*"


def use_manual_mask():
    if STATE.get("manual_mask_u8") is None:
        return _mask_source_text(), None, None
    STATE["active_mask_source"] = "manual"
    m = STATE["manual_mask_u8"]
    vis = overlay_mask(STATE["working_np"], m) if STATE.get("working_np") is not None else None
    return _mask_source_text(), (Image.fromarray(vis) if vis is not None else None), Image.fromarray(m)


def use_auto_mask():
    if STATE.get("auto_mask_u8") is None:
        return _mask_source_text(), None, None
    STATE["active_mask_source"] = "auto"
    m = STATE["auto_mask_u8"]
    vis = overlay_mask(STATE["working_np"], m) if STATE.get("working_np") is not None else None
    return _mask_source_text(), (Image.fromarray(vis) if vis is not None else None), Image.fromarray(m)


def clear_mask():
    STATE["manual_mask_u8"] = None
    STATE["auto_mask_u8"] = None
    STATE["active_mask_source"] = None
    STATE["selected_mask"] = None
    STATE["auto_mask_candidates"] = []
    if STATE["working_np"] is None:
        return None, None, None, "Cleared."
    return Image.fromarray(STATE["working_np"]), None, None, "Mask cleared."

def apply_inpaint(
    prompt: str,
    negative: str,
    steps: int,
    strength: float,
    guidance: float,
    expand_px: int,
    blur_px: int,
    seed: int,
    auto_enrich: bool,
    edit_mode: str,
    use_controlnet: bool = False,
    controlnet_type: str = "depth",
    do_refine: bool = False,
):
    global pipe, PIPE, controlnet_pipes, img2img_pipe

    t0 = time.time()

    used_seed_str = str(seed if seed is not None and int(seed) >= 0 else "random")
    # Realtime stage indicator (Gradio streaming)
    def _pack(stage_md: str, run_msg: str = '', final_prompt: str = ''):
        # Keep the UI responsive by yielding intermediate tuples.
        # outputs: [output, run_status, positive_final_preview, global_status, seed_display]
        # show stage in run_status during execution
        stage_line = stage_md.replace('### Stage: ', '').strip()
        return None, (run_msg or stage_line), final_prompt, stage_md, used_seed_str

    yield _pack('### Stage: Preparing inputs')
    if MOCK_INPAINT:
        yield _pack("### Stage: MOCK_INPAINT", "MOCK_INPAINT mode")
        return

    # Public demo guardrails are applied via UI + server-side clamps below.

    # Seed 기반 generator 생성
    gen = None
    if seed is not None and int(seed) >= 0:
        gen = torch.Generator(DEVICE).manual_seed(int(seed))

    # Pick active mask source
    mask_u8 = _get_active_mask()

    if mask_u8 is None:
        yield _pack("### Stage: Error", "Mask missing")
        return

    # 마스크 postprocess
    mask_pp = postprocess_mask(mask_u8, int(expand_px), int(blur_px))
    mask_pil = Image.fromarray(mask_pp).convert("L")
    image_pil = STATE["working_pil"].convert("RGB")

    fin = build_final_prompts(prompt, negative, auto_enrich, edit_mode)
    positive_final = fin["prompt"]
    negative_final = fin["negative"]
    prompt_2 = fin["prompt_2"]
    negative_2 = fin["negative_2"]

    if fin["warnings"]:
        for w in fin["warnings"]:
            print(f"[PROMPT][WARN] {w}")
    yield _pack("### Stage: Prompt ready", final_prompt=positive_final)

    # Mode-specific clamps can be added here if needed.

    steps = int(steps)
    strength = float(strength)
    guidance = float(guidance)

    # Guardrails for public demo stability
    if PUBLIC_DEMO:
        steps = max(1, min(int(steps), PUBLIC_MAX_STEPS))

    # --- 간단 병목 프로파일링(로그용) ---
    prof = {}
    def mark(k):
        prof[k] = time.time()

    mark("start")

    result = None
    refined = None

    print(f"[START] Generation start | mode={edit_mode} | controlnet={use_controlnet} | steps={steps} | seed={used_seed_str}")

    # ControlNet 분기
    if use_controlnet:
        if controlnet_type == "depth":
            repo = CONTROLNET_DEPTH
        elif controlnet_type == "openpose":
            repo = CONTROLNET_OPENPOSE
        else:
            repo = "lllyasviel/control_v11p_sd15_inpaint"

        # NOTE: key에 strength 넣으면 캐시가 과하게 늘어남 → type만으로 캐시
        key = f"{controlnet_type}"

        if key not in controlnet_pipes:
            yield _pack(f"### Stage: Loading ControlNet ({controlnet_type})")
            mark("cn_load_start")
            try:
                controlnet = ControlNetModel.from_pretrained(
                    repo,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                    local_files_only=True
                )
                controlnet_pipes[key] = StableDiffusionXLControlNetInpaintPipeline.from_single_file(
                    JUGGERNAUT_INPAINT,
                    controlnet=controlnet,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                controlnet_pipes[key].to(DEVICE)
                print(f"[CONTROLNET] Loaded successfully on {DEVICE}")
            except Exception as e:
                print(f"[CONTROLNET] Load failed: {type(e).__name__}: {str(e)}")
                use_controlnet = False
            mark("cn_load_end")

        if use_controlnet and key in controlnet_pipes:
            p = controlnet_pipes[key]
            print(f"[CONTROLNET] Using {controlnet_type} | type={type(p).__name__}")
            yield _pack(f"### Stage: Running ControlNet ({controlnet_type})")
            mark("cn_run_start")
            try:
                result = p(
                    prompt=positive_final,
                    prompt_2=prompt_2,
                    negative_prompt=negative_final,
                    negative_prompt_2=negative_2,
                    image=image_pil,
                    mask_image=mask_pil,
                    control_image=image_pil,
                    controlnet_conditioning_scale=0.65,
                    num_inference_steps=steps,
                    strength=strength,
                    guidance_scale=guidance,
                    generator=gen,
                ).images[0]
                print("[CONTROLNET] Generation success!")
            except Exception as e:
                print(f"[CONTROLNET] Generation failed: {str(e)} → fallback to base")
                result = None
            mark("cn_run_end")

    # 기본 Inpaint
    if result is None:
        if pipe is None:
            load_pipe()
        yield _pack("### Stage: Running Inpaint")
        mark("base_run_start")
        try:
            result = pipe(
                prompt=positive_final,
                prompt_2=prompt_2,
                negative_prompt=negative_final,
                negative_prompt_2=negative_2,
                image=image_pil,
                mask_image=mask_pil,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=guidance,
                generator=gen,
            ).images[0]
            print("[INPAINT] Generation success!")
        except Exception as e:
            err = str(e)
            print(f"[INPAINT] Base generation failed: {err}")

            # Best-effort OOM recovery for consecutive runs
            if "out of memory" in err.lower() or "cuda" in err.lower() and "memory" in err.lower():
                print("[OOM] Attempting recovery: unload aux + hard clear")
                try:
                    unload_aux_pipelines()
                except Exception:
                    pass
                try:
                    hard_clear_vram()
                except Exception:
                    pass
                return (
                    None,
                    "CUDA OOM: VRAM 부족으로 실패했습니다. Hard Clear를 수행했습니다. 해상도/steps를 낮추거나 LOW_VRAM=1, CPU_OFFLOAD=1을 시도해보세요.",
                    positive_final,
                    "Error: CUDA OOM (recovered)",
                    used_seed_str,
                )

            return None, f"Generation failed: {err}", positive_final, "Error", used_seed_str
        mark("base_run_end")

    # Refine pass
    if do_refine and result is not None:
        yield _pack("### Stage: Loading/Running Refine")
        print("[REFINE] Refine pass 시작")
        mark("refine_load_start")
        try:
            if img2img_pipe is None:
                model_path = JUGGERNAUT_INPAINT if os.path.exists(JUGGERNAUT_INPAINT) else DEFAULT_MODEL
                img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
                    model_path,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    use_safetensors=True,
                )
                img2img_pipe.to(DEVICE)
                print("[REFINE] Img2Img pipeline loaded")
            mark("refine_load_end")

            mark("refine_run_start")
            refined = img2img_pipe(
                prompt=positive_final,
                prompt_2=prompt_2,
                image=result,
                strength=0.20,
                num_inference_steps=10,
                guidance_scale=guidance,
                generator=gen,
            ).images[0]
            mark("refine_run_end")
            print("[REFINE] Refine completed")
        except Exception as e:
            print(f"[REFINE] Failed: {str(e)} → using first result")
            refined = result

    final_image = refined if do_refine and refined is not None else result

    dt = time.time() - t0

    # 프로파일 로그(대략 어디서 막히는지)
    def _dur(a, b):
        if a in prof and b in prof:
            return prof[b] - prof[a]
        return None

    prof_msg_parts = []
    d_cn_load = _dur("cn_load_start", "cn_load_end")
    d_cn_run = _dur("cn_run_start", "cn_run_end")
    d_base = _dur("base_run_start", "base_run_end")
    d_ref_load = _dur("refine_load_start", "refine_load_end")
    d_ref_run = _dur("refine_run_start", "refine_run_end")

    if d_cn_load is not None: prof_msg_parts.append(f"cn_load:{d_cn_load:.2f}s")
    if d_cn_run is not None:  prof_msg_parts.append(f"cn_run:{d_cn_run:.2f}s")
    if d_base is not None:    prof_msg_parts.append(f"base_run:{d_base:.2f}s")
    if d_ref_load is not None:prof_msg_parts.append(f"ref_load:{d_ref_load:.2f}s")
    if d_ref_run is not None: prof_msg_parts.append(f"ref_run:{d_ref_run:.2f}s")
    prof_msg = " | ".join(prof_msg_parts) if prof_msg_parts else "profile:n/a"

    # Human-friendly performance breakdown + bottleneck guess
    stages = []
    if d_cn_load is not None: stages.append(("ControlNet load", d_cn_load))
    if d_cn_run is not None:  stages.append(("ControlNet run", d_cn_run))
    if d_base is not None:    stages.append(("Inpaint run", d_base))
    if d_ref_load is not None:stages.append(("Refine load", d_ref_load))
    if d_ref_run is not None: stages.append(("Refine run", d_ref_run))

    bottleneck_name = None
    bottleneck_reason = None
    bottleneck_tip = None
    if stages:
        bottleneck_name, bottleneck_val = max(stages, key=lambda x: x[1])
        if bottleneck_name in ("ControlNet load", "Refine load"):
            bottleneck_reason = "Model pipeline had to be loaded into memory (first run or after unload)."
            bottleneck_tip = "Tip: keep the option enabled between runs, or disable AUTO_UNLOAD_AUX if you prefer speed over VRAM stability."
        elif bottleneck_name in ("ControlNet run",):
            bottleneck_reason = "ControlNet adds extra compute and increases VRAM pressure."
            bottleneck_tip = "Tip: try fewer steps / smaller Working Long Side, or disable ControlNet for faster runs."
        elif bottleneck_name in ("Refine run",):
            bottleneck_reason = "Refine is an additional img2img pass (extra inference)."
            bottleneck_tip = "Tip: disable Refine, or lower steps/size if you hit memory/time limits."
        else:
            bottleneck_reason = "Most time is spent in SDXL inference (steps × resolution)."
            bottleneck_tip = "Tip: lower Steps or Working Long Side; LOW_VRAM=1 improves stability but may reduce speed."

    model_info = f"Juggernaut XL | ControlNet: {use_controlnet} ({controlnet_type}) | Refine: {do_refine}"
    run_msg = f"완료! {model_info} | Time: {int(dt // 60)}m {int(dt % 60)}s"

    # Markdown-friendly status block
    perf_lines = []
    for name, sec in stages:
        perf_lines.append(f"- {name}: **{sec:.2f}s**")
    perf_md = "\n".join(perf_lines) if perf_lines else "- (n/a)"

    bottleneck_md = ""
    if bottleneck_name:
        bottleneck_md = (
            f"\n### Bottleneck (best guess)\n"
            f"- **{bottleneck_name}**\n"
            f"- Why: {bottleneck_reason}\n"
            f"- {bottleneck_tip}\n"
        )

    global_status = (
        f"### Done\n"
        f"- Mode: **{edit_mode}**\n"
        f"- Steps: **{steps}** | Strength: **{strength:.2f}** | CFG: **{guidance:.1f}**\n"
        f"- ControlNet: **{use_controlnet}** ({controlnet_type})\n"
        f"- Refine: **{do_refine}**\n"
        f"- Seed: **{used_seed_str}**\n\n"
        f"### Timing\n"
        f"- Total: **{int(dt // 60)}m {int(dt % 60)}s**\n"
        f"{perf_md}\n"
        f"\n_Profile raw_: `{prof_msg}`\n"
        f"{bottleneck_md}"
    )

    # Post-run VRAM housekeeping
    if AUTO_UNLOAD_AUX:
        unload_aux_pipelines()
        print("[VRAM] AUTO_UNLOAD_AUX: unloaded ControlNet/Refine pipelines")

    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception:
            pass
        print("[VRAM] Cleared cache after generation")

    # Free large references from auto-mask gallery to reduce RAM usage on next runs
    try:
        STATE['auto_mask_candidates'] = []
        gc.collect()
    except Exception:
        pass

    # seed 기록
    try:
        with open("last_seed.txt", "w") as f:
            f.write(used_seed_str)
    except Exception:
        pass

    # Auto hard clear if memory pressure is extreme (helps consecutive runs)
    if DEVICE == "cuda" and torch.cuda.is_available():
        try:
            dev = torch.cuda.current_device()
            resv = torch.cuda.memory_reserved(dev)
            total = torch.cuda.get_device_properties(dev).total_memory
            ratio = float(resv) / float(total) if total else 0.0
            if ratio >= AUTO_HARD_CLEAR_THRESHOLD:
                print(f"[VRAM][AUTO] reserved/total={ratio:.2f} >= {AUTO_HARD_CLEAR_THRESHOLD:.2f} → hard clear")
                hard_clear_vram()
        except Exception:
            pass

    # VRAM is displayed in the dedicated VRAM box; keep status focused.

    yield final_image, run_msg, positive_final, global_status, used_seed_str
    return

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

CSS = """
:root{
  --radius: 14px;
  --border: rgba(0,0,0,0.10);
}
.gradio-container { max-width: 1280px !important; }
"""

# (Public repo) Admin unlock helpers removed.

def build_ui():
    TEXT = {
        "en": {
            "title": "## ImageAI Demo (SDXL Inpaint)",
            "help_title": "Help / Quick Guide",
            "help": (
                "1) **Upload** an image\n"
                "2) Make a **Mask**\n"
                "   - Auto Mask: click **Auto Mask (v5)** (requires MediaPipe model)\n"
                "   - Manual Mask: click on the image (requires SAM weights)\n"
                "3) Write **Positive / Negative** prompts\n"
                "4) (Optional) enable **ControlNet** or **Refine** (uses more VRAM)\n"
                "5) Click **Apply**\n\n"
                "If the 2nd run fails with memory errors:\n"
                "- Lower **Working Long Side** / **Steps**\n"
                "- Click **Unload Aux** (recommended) or **Hard Clear**\n"
                "- In `.env`: `LOW_VRAM=1`, `AUTO_UNLOAD_AUX=1`\n"
            ),
            "lang": "Language",
            "status": "Status / Logs",
            "vram": "GPU VRAM",
            "result": "Result",
            "run_status": "Run status",
            "working": "Working",
            "mask": "Mask (selected)",
            "auto_mask": "Auto Mask",
            "prompt": "Prompt",
            "settings": "Settings",
            "auto_enrich": "Auto-enrich prompt",
            "pos": "Positive Prompt",
            "neg": "Negative Prompt",
            "prompt_check": "Prompt Check",
            "prompt_check_label": "Prompt Check (final prompts + token counts)",
            "final_prompt": "Final prompt (applied)",
            "apply": "Apply",
            "clear_mask": "Clear Mask",
            "auto_mask_btn": "Auto Mask (v5)",
            "working_long": "Working Long Side (px)",
            "sam": "SAM Model (Manual Mask)",
            "mask_expand": "Mask Expand (px)",
            "mask_blur": "Mask Blur (px)",
            "steps": "Inference Steps",
            "strength": "Strength",
            "guidance": "Guidance Scale (CFG)",
            "seed": "Seed (-1 = random)",
            "used_seed": "Used Seed",
            "use_cn": "ControlNet",
            "cn_type": "Type",
            "refine": "Refine Pass",
        },
        "kr": {
            "title": "## ImageAI 데모 (SDXL 인페인트)",
            "help_title": "도움말 / 빠른 가이드",
            "help": (
                "1) 이미지 **업로드**\n"
                "2) **마스크** 만들기\n"
                "   - Auto Mask(v5): MediaPipe 모델 필요\n"
                "   - Manual Mask: 이미지 클릭(SAM weights 필요)\n"
                "3) **Positive/Negative** 프롬프트 입력\n"
                "4) (선택) **ControlNet / Refine** (VRAM 추가 사용)\n"
                "5) **Apply** 클릭\n\n"
                "두 번째 실행에서 메모리 에러가 나면:\n"
                "- Working Long Side / Steps 낮추기\n"
                "- **Unload Aux** 또는 **Hard Clear**\n"
                "- `.env`: `LOW_VRAM=1`, `AUTO_UNLOAD_AUX=1`\n"
            ),
            "lang": "언어",
            "status": "상태 / 로그",
            "vram": "GPU VRAM",
            "result": "결과",
            "run_status": "실행 상태",
            "working": "작업 이미지",
            "mask": "마스크(선택)",
            "auto_mask": "자동 마스크",
            "prompt": "프롬프트",
            "settings": "설정",
            "auto_enrich": "프롬프트 자동 확장",
            "pos": "Positive Prompt",
            "neg": "Negative Prompt",
            "prompt_check": "프롬프트 체크",
            "prompt_check_label": "프롬프트 체크(최종 prompt/prompt_2 + 토큰 수)",
            "final_prompt": "최종 prompt(적용됨)",
            "apply": "적용(Apply)",
            "clear_mask": "마스크 지우기",
            "auto_mask_btn": "Auto Mask (v5)",
            "working_long": "작업 해상도(긴 변 px)",
            "sam": "SAM 모델(수동 마스크)",
            "mask_expand": "마스크 확장(px)",
            "mask_blur": "마스크 블러(px)",
            "steps": "스텝(steps)",
            "strength": "강도(strength)",
            "guidance": "CFG(guidance)",
            "seed": "시드(-1 랜덤)",
            "used_seed": "사용된 시드",
            "use_cn": "ControlNet",
            "cn_type": "타입",
            "refine": "리파인(Refine)",
        },
    }

    def t(lang: str, k: str) -> str:
        lang = lang if lang in TEXT else "en"
        return TEXT[lang].get(k, TEXT["en"].get(k, k))

    def render_help(lang: str):
        return t(lang, "help")

    with gr.Blocks(title="ImageAI Demo") as demo:
        lang = gr.Dropdown(["en", "kr"], value="en", label="Language")
        title_md = gr.Markdown(t("en", "title"))

        with gr.Accordion(t("en", "help_title"), open=True):
            help_md = gr.Markdown(render_help("en"))

        # Top bar: Status + VRAM + buttons
        with gr.Row():
            with gr.Column(scale=7):
                global_status = gr.Markdown(value="**Ready.**")
            with gr.Column(scale=5):
                vram_box = gr.Textbox(label=t("en", "vram"), value=get_vram_text(), lines=7, interactive=False)
                with gr.Row():
                    btn_vram_refresh = gr.Button("VRAM Refresh")
                    btn_soft_clear = gr.Button("🧹 Soft Clear")
                    btn_unload_aux = gr.Button("Unload Aux")
                    btn_hard_clear = gr.Button("🔥 Hard Clear")

        # Main layout: Result on top-left, Tabs on right
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    gr.Markdown(f"### {t('en','result')}")
                    output = gr.Image(height=520)
                    run_status = gr.Textbox(lines=2, label=t("en", "run_status"))

                with gr.Group():
                    gr.Markdown(f"### {t('en','working')}")
                    input_image = gr.Image(type="pil", height=420)

                with gr.Group():
                    gr.Markdown(f"### {t('en','mask')}")
                    mask_overlay = gr.Image(type="numpy", height=220)
                    selected_mask_preview = gr.Image(type="numpy", height=300)

            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("Mask"):
                        gr.Markdown(f"### {t('en','auto_mask')}")
                        auto_target = gr.Dropdown(["top", "pants", "hair", "background", "person", "auto"], value="top", label="Auto mask target")
                        auto_gallery = gr.Gallery(columns=4, height=320)
                        auto_status = gr.Textbox(lines=2)
                        with gr.Row():
                            btn_auto = gr.Button(t("en", "auto_mask_btn"))
                            btn_clear = gr.Button(t("en", "clear_mask"))

                        active_mask_md = gr.Markdown(_mask_source_text())
                        with gr.Row():
                            btn_use_manual = gr.Button("Use Manual")
                            btn_use_auto = gr.Button("Use Auto")

                    with gr.TabItem("Prompt"):
                        auto_enrich = gr.Checkbox(value=True, label=t("en", "auto_enrich"))
                        edit_mode = gr.Dropdown(choices=get_edit_mode_choices(), value="Wear / Change Clothes")
                        prompt = gr.Textbox(lines=3, label=t("en", "pos"))
                        negative = gr.Textbox(lines=3, label=t("en", "neg"))
                        preview_btn = gr.Button(t("en", "prompt_check"), variant="secondary")
                        preview_output = gr.Markdown(value="")
                        positive_final_preview = gr.Textbox(lines=3, interactive=False, label=t("en", "final_prompt"))

                    with gr.TabItem("Settings"):
                        # Performance + VRAM settings are always visible in public
                        if PUBLIC_DEMO:
                            working_long_side = gr.Slider(512, PUBLIC_MAX_LONG_SIDE, value=min(896, PUBLIC_MAX_LONG_SIDE), step=64, label=t("en", "working_long"))
                            steps = gr.Slider(10, PUBLIC_MAX_STEPS, value=min(18, PUBLIC_MAX_STEPS), label=t("en", "steps"))
                        else:
                            working_long_side = gr.Slider(512, 1536, value=1024, step=64, label=t("en", "working_long"))
                            steps = gr.Slider(10, 60, value=28, label=t("en", "steps"))

                        sam_model = gr.Dropdown(["vit_b", "vit_h"], value="vit_b", label=t("en", "sam"))
                        mask_expand = gr.Slider(0, 40, value=10, label=t("en", "mask_expand"))
                        mask_blur = gr.Slider(0, 40, value=18, label=t("en", "mask_blur"))
                        strength = gr.Slider(0.3, 0.95, value=0.55, label=t("en", "strength"))
                        guidance = gr.Slider(1.0, 12.0, value=7.0, label=t("en", "guidance"))
                        seed = gr.Number(value=-1, label=t("en", "seed"))
                        seed_display = gr.Textbox(label=t("en", "used_seed"), interactive=False)

                        with gr.Row():
                            use_controlnet = gr.Checkbox(label=t("en", "use_cn"), value=False)
                            controlnet_type = gr.Dropdown(["depth", "openpose", "inpaint"], value="depth", label=t("en", "cn_type"))

                        do_refine = gr.Checkbox(label=t("en", "refine"), value=False, interactive=True)
                        btn_apply = gr.Button(t("en", "apply"), variant="primary")

        # Events
        input_image.upload(fn=on_upload, inputs=[input_image, working_long_side], outputs=[input_image, selected_mask_preview, auto_status, global_status])
        input_image.select(fn=on_manual_click, inputs=[sam_model], outputs=[mask_overlay, selected_mask_preview, auto_status, global_status])
        input_image.select(fn=lambda: (_mask_source_text(),), inputs=None, outputs=[active_mask_md])

        btn_auto.click(fn=build_auto_candidates_v5, inputs=[prompt, auto_enrich, edit_mode, auto_target], outputs=[auto_gallery, auto_status, positive_final_preview])
        btn_auto.click(fn=use_auto_mask, inputs=None, outputs=[active_mask_md, mask_overlay, selected_mask_preview])
        btn_clear.click(fn=clear_mask, outputs=[mask_overlay, selected_mask_preview, auto_gallery, auto_status])
        btn_clear.click(fn=lambda: (_mask_source_text(),), inputs=None, outputs=[active_mask_md])

        preview_btn.click(fn=preview_enriched_prompt, inputs=[prompt, negative, auto_enrich, edit_mode], outputs=[preview_output])
        preview_btn.click(fn=lambda p,n,a,m: (build_final_prompts(p,n,a,m)["prompt"],), inputs=[prompt, negative, auto_enrich, edit_mode], outputs=[positive_final_preview])

        btn_apply.click(
            fn=apply_inpaint,
            inputs=[
                prompt, negative, steps, strength, guidance,
                mask_expand, mask_blur, seed, auto_enrich, edit_mode,
                use_controlnet, controlnet_type,
                do_refine,
            ],
            outputs=[output, run_status, positive_final_preview, global_status, seed_display],
        )
        btn_apply.click(fn=lambda: (get_vram_text(),), inputs=None, outputs=[vram_box])

        btn_use_manual.click(fn=use_manual_mask, inputs=None, outputs=[active_mask_md, mask_overlay, selected_mask_preview])
        btn_use_auto.click(fn=use_auto_mask, inputs=None, outputs=[active_mask_md, mask_overlay, selected_mask_preview])

        # VRAM buttons
        btn_vram_refresh.click(fn=lambda: (get_vram_text(),), inputs=None, outputs=[vram_box])
        btn_soft_clear.click(fn=soft_clear_vram, inputs=None, outputs=[global_status, vram_box])
        btn_unload_aux.click(fn=lambda: (unload_aux_pipelines() or "[VRAM] Unloaded aux pipelines.", get_vram_text()), inputs=None, outputs=[global_status, vram_box])
        btn_hard_clear.click(fn=hard_clear_vram, inputs=None, outputs=[global_status, vram_box])

        # Language switching (best-effort: update help/title; some labels are static)
        def _on_lang_change(l):
            return (
                t(l, "title"),
                render_help(l),
            )

        lang.change(fn=_on_lang_change, inputs=[lang], outputs=[title_md, help_md])

    return demo

def _queue_compat(app: gr.Blocks, max_size: int, concurrency: int):
    """Call gradio queue() with compatible kwargs across versions."""
    import inspect

    try:
        sig = inspect.signature(app.queue)
        params = set(sig.parameters.keys())
    except Exception:
        params = set()

    kwargs = {}
    if "max_size" in params:
        kwargs["max_size"] = max_size

    # Gradio queue concurrency kwarg differs by version.
    if "concurrency_count" in params:
        kwargs["concurrency_count"] = concurrency
    elif "default_concurrency_limit" in params:
        kwargs["default_concurrency_limit"] = concurrency
    elif "concurrency_limit" in params:
        kwargs["concurrency_limit"] = concurrency

    try:
        return app.queue(**kwargs)
    except TypeError:
        # Last resort: call without kwargs
        return app.queue()


if __name__ == "__main__":
    demo = build_ui()
    # Queue settings
    if PUBLIC_DEMO:
        _queue_compat(demo, max_size=PUBLIC_MAX_QUEUE, concurrency=PUBLIC_CONCURRENCY)
    else:
        _queue_compat(demo, max_size=10, concurrency=1)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        debug=True,
        share=False,
        prevent_thread_lock=True,
        css=CSS,
        theme=theme
    )
