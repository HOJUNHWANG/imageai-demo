# Setup Notes

## Quick start (UI only)
If you want to verify the UI without downloading large models:

```bash
set MOCK_INPAINT=1
python app.py
```

## MediaPipe model file
Auto-mask uses MediaPipe Tasks segmentation and expects:

- `weights/selfie_multiclass_256x256.tflite`

If the file is missing, the app will warn on boot and auto-mask will be disabled.

## SAM weights
Manual click-to-mask uses SAM and expects one of:
- `weights/sam_vit_b_01ec64.pth`
- `weights/sam_vit_h_4b8939.pth`

## ControlNet folders (optional)
If you want to enable ControlNet with local files, place them under:
- `models/ControlNet/controlnet-depth-sdxl-1.0`
- `models/ControlNet/controlnet-openpose-sdxl-1.0`
