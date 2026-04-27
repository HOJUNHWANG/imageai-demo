"""Environment checker for ImageAI Studio.

Run:
  python scripts/doctor.py

Checks that required models and dependencies are available.
"""
from __future__ import annotations

import os
import importlib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
MODELS_DIR = os.path.join(BASE_DIR, "models", "stable-diffusion-xl")

FILE_CHECKS = {
    "MediaPipe tflite": os.path.join(WEIGHTS_DIR, "selfie_multiclass_256x256.tflite"),
}

OPTIONAL_FILES = {
    "JuggernautXL checkpoint": os.path.join(MODELS_DIR, "juggernautXL_ragnarokBy.safetensors"),
}

REQUIRED_PACKAGES = [
    "torch", "diffusers", "transformers", "accelerate",
    "fastapi", "uvicorn", "PIL", "numpy", "mediapipe",
]

OPTIONAL_PACKAGES = [
    ("bitsandbytes", "NF4 quantization for FLUX"),
    ("xformers", "Memory-efficient attention"),
    ("cv2", "OpenCV (mask post-processing)"),
]


def main() -> None:
    print("ImageAI Studio — Environment Check\n")
    issues = 0

    # Required files
    print("── Required Files ──")
    for name, path in FILE_CHECKS.items():
        ok = os.path.exists(path)
        print(f"  {'✓' if ok else '✗'} {name}: {path}")
        if not ok:
            issues += 1

    # Optional files
    print("\n── Optional Files ──")
    for name, path in OPTIONAL_FILES.items():
        ok = os.path.exists(path)
        print(f"  {'✓' if ok else '–'} {name}: {path}")

    # Required packages
    print("\n── Required Packages ──")
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} — MISSING")
            issues += 1

    # Optional packages
    print("\n── Optional Packages ──")
    for pkg, desc in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            print(f"  ✓ {pkg} — {desc}")
        except ImportError:
            print(f"  – {pkg} — {desc} (not installed)")

    # GPU check
    print("\n── GPU ──")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  ✓ {name} ({mem:.1f} GB)")
        else:
            print("  – No CUDA GPU detected (CPU mode)")
    except Exception as e:
        print(f"  ✗ GPU check failed: {e}")

    print()
    if issues:
        print(f"⚠ {issues} issue(s) found. Fix them before running.")
        raise SystemExit(1)
    else:
        print("✓ All checks passed.")


if __name__ == "__main__":
    main()
