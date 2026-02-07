# Demo Walkthrough

This repo is a **portfolio-friendly local demo** for SDXL inpainting.

## 0) Quick smoke test (no models)
If you just want to confirm the UI works without downloading large weights:

### Windows (cmd / PowerShell)
```bash
set MOCK_INPAINT=1
python app.py
```
Open: http://127.0.0.1:7860

What to check:
- UI loads
- Upload works
- Mask tools render
- Prompt Check runs
- Apply returns a mock result/status (no crash)

## 1) Full demo (with models)
Prereqs (high level):
- Python 3.11
- CUDA GPU recommended
- Put required files under `weights/` and optional models under `models/`

See: `docs/SETUP.md`

Recommended public-stable settings:
- `PUBLIC_DEMO=1`
- `LOW_VRAM=1`
- `AUTO_UNLOAD_AUX=1`
- (if still OOM) `CPU_OFFLOAD=1`

Workflow:
1) Upload an image
2) Create a mask
   - Auto Mask(v5) if MediaPipe model is present
   - Manual Mask by clicking if SAM weights are present
3) Write prompt / negative
4) Click **Prompt Check** and confirm the final prompt + token counts
5) (Optional) enable ControlNet / Refine
6) Click **Apply**

After run:
- Check **Timing** section (which stage is slow)
- If 2nd run fails, click **Unload Aux** or **Hard Clear**
