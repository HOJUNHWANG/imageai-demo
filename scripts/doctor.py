"""Simple environment checker for ImageAI Demo.

Run:
  python scripts/doctor.py

This does not download anything; it only prints what is missing.
"""

from __future__ import annotations

import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

CHECKS = {
    "MediaPipe tflite": os.path.join(WEIGHTS_DIR, "selfie_multiclass_256x256.tflite"),
    "SAM vit_b": os.path.join(WEIGHTS_DIR, "sam_vit_b_01ec64.pth"),
    "SAM vit_h": os.path.join(WEIGHTS_DIR, "sam_vit_h_4b8939.pth"),
}


def main() -> None:
    print("ImageAI Demo doctor\n")
    missing = []
    for name, path in CHECKS.items():
        ok = os.path.exists(path)
        print(f"- {name}: {'OK' if ok else 'MISSING'} ({path})")
        if not ok:
            missing.append(name)

    print("\nTip: You can still start the UI with MOCK_INPAINT=1.")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
