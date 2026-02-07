# Architecture Notes

## Goal
A local SDXL inpainting demo with **mask-first UX** and practical stability guardrails.

## Components
- **UI**: Gradio
- **Inpaint**: `StableDiffusionXLInpaintPipeline`
- **(Optional) ControlNet**: `StableDiffusionXLControlNetInpaintPipeline`
- **(Optional) Refine**: `StableDiffusionXLImg2ImgPipeline`
- **Manual mask**: SAM click-to-mask (requires `weights/*.pth`)
- **Auto mask**: MediaPipe Tasks segmentation (requires `weights/selfie_multiclass_256x256.tflite`)

## Memory stability strategy (public demo)
Typical failure mode: consecutive runs accumulate VRAM/RAM pressure.

Mitigations:
- Public guardrails: max long side / max steps / queue limits
- `LOW_VRAM`: attention slicing + VAE slicing/tiling
- `AUTO_UNLOAD_AUX`: unload ControlNet + Refine pipelines after each run
- UI tools: Soft Clear / Unload Aux / Hard Clear
- Prompt safety: SDXL 77-token overflow split to `prompt_2` / `negative_prompt_2` + safe truncation

## Performance observability
After each Apply:
- stage-level timing breakdown (e.g., inpaint run, controlnet run, refine run)
- best-effort bottleneck explanation + user tips
- optional realtime stage indicator during Apply
