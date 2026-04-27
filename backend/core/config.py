"""
Shared runtime configuration for ImageAI Studio backend.
Extracted from app.py globals.
"""
import os
import torch

# Memory behavior
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if os.getenv("DEBUG_CUDA", "0") == "1":
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# Speed boost on RTX 30/40
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # imageAI_public/
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODELS_DIR = os.path.join(BASE_DIR, "models", "stable-diffusion-xl")

JUGGERNAUT_INPAINT = os.path.join(MODELS_DIR, "juggernautXL_ragnarokBy.safetensors")
DEFAULT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"

CONTROLNET_DEPTH = "diffusers/controlnet-depth-sdxl-1.0"
CONTROLNET_OPENPOSE = "thibaud/controlnet-openpose-sdxl-1.0"
CONTROLNET_CANNY = "diffusers/controlnet-canny-sdxl-1.0"

FLUX_FILL_MODEL = os.getenv("FLUX_FILL_MODEL", "black-forest-labs/FLUX.1-Fill-dev")
FLUX_KONTEXT_MODEL = os.getenv("FLUX_KONTEXT_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")

MOCK_INPAINT = os.getenv("MOCK_INPAINT", "0") == "1"

# torch.compile for SDXL UNet — 20-40% speedup after first run.
# First load adds ~60-120s compilation time. Requires PyTorch 2.0+.
COMPILE_UNET = os.getenv("COMPILE_UNET", "0") == "1"

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        print(f"[CONFIG] Warning: invalid value for {key}, using default {default}")
        return default

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        print(f"[CONFIG] Warning: invalid value for {key}, using default {default}")
        return default

# Public demo flags
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "1") == "1"
PUBLIC_MAX_LONG_SIDE = _env_int("PUBLIC_MAX_LONG_SIDE", 896)
PUBLIC_MAX_STEPS = _env_int("PUBLIC_MAX_STEPS", 22)
PUBLIC_MAX_QUEUE = _env_int("PUBLIC_MAX_QUEUE", 10)
PUBLIC_CONCURRENCY = _env_int("PUBLIC_CONCURRENCY", 1)

# VRAM/RAM stability toggles
LOW_VRAM = os.getenv("LOW_VRAM", "1" if PUBLIC_DEMO else "0") == "1"
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"
AUTO_UNLOAD_AUX = os.getenv("AUTO_UNLOAD_AUX", "1" if PUBLIC_DEMO else "0") == "1"
AUTO_HARD_CLEAR_THRESHOLD = _env_float("AUTO_HARD_CLEAR_THRESHOLD", 0.92)

# Device
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
torch.set_float32_matmul_precision("high")

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}")
