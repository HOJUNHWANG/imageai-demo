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

MOCK_INPAINT = os.getenv("MOCK_INPAINT", "0") == "1"

# Public demo flags
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "1") == "1"
PUBLIC_MAX_LONG_SIDE = int(os.getenv("PUBLIC_MAX_LONG_SIDE", "896"))
PUBLIC_MAX_STEPS = int(os.getenv("PUBLIC_MAX_STEPS", "22"))
PUBLIC_MAX_QUEUE = int(os.getenv("PUBLIC_MAX_QUEUE", "10"))
PUBLIC_CONCURRENCY = int(os.getenv("PUBLIC_CONCURRENCY", "1"))

# VRAM/RAM stability toggles
LOW_VRAM = os.getenv("LOW_VRAM", "1" if PUBLIC_DEMO else "0") == "1"
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "0") == "1"
AUTO_UNLOAD_AUX = os.getenv("AUTO_UNLOAD_AUX", "1" if PUBLIC_DEMO else "0") == "1"
AUTO_HARD_CLEAR_THRESHOLD = float(os.getenv("AUTO_HARD_CLEAR_THRESHOLD", "0.92"))

# Device
def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = pick_device()
torch.set_float32_matmul_precision("high")

print(f"DEVICE={DEVICE}, MOCK_INPAINT={MOCK_INPAINT}")
