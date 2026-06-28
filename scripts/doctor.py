"""Check the local studio without downloading model weights."""
from __future__ import annotations

import importlib
import platform
import sys


REQUIRED = (
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "xformers",
    "fastapi",
    "PIL",
    "psutil",
)


def main() -> None:
    print("Morrow local environment\n")
    print(f"Python: {sys.version.split()[0]} ({platform.system()})")
    missing: list[str] = []
    for package in REQUIRED:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "installed")
            print(f"  OK  {package} {version}")
        except Exception as exc:
            missing.append(package)
            print(f"  --  {package}: {exc}")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"\nGPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
            print(f"CUDA: {torch.version.cuda}")
        else:
            print("\nGPU: CUDA unavailable")
    except Exception:
        pass

    if missing:
        raise SystemExit(f"\nMissing: {', '.join(missing)}")
    print("\nEnvironment is ready. Models download lazily on first use.")


if __name__ == "__main__":
    main()
