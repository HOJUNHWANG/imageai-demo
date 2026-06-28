# Architecture

Morrow is a local-only image generation and instruction editing studio designed
around one 12 GB GPU.

## Runtime

- FastAPI owns one process-wide inference lock. A competing request receives
  `409` instead of racing for VRAM.
- The model cache key is `(workflow, profile)`. An identical next request reuses
  the warm pipeline; changing either value unloads it, collects Python objects,
  and clears CUDA caches.
- Large transformers and text encoders use NF4 where appropriate. The Balanced
  Z-Image repository is already NF4 and is never re-quantized.
- Torch 2.6/CUDA 12.4 uses pinned xFormers 0.0.29.post2. The loader requests its
  attention backend and automatically retains native SDPA if unavailable.
- Model CPU offload, TF32, VAE slicing/tiling, and per-profile pixel budgets are
  enabled for the RTX 3080 Ti.
- Results expose load, inference, and post-processing timings so model download,
  PCIe/offload, denoising, and PNG/compositing delays are distinguishable.

## Profiles

Generation selects between a 40-step FLUX model, an 8-step pre-quantized NSFW
Z-Image model, and an 8-step Z-Image model with a heretic text encoder. Editing
selects between the original 40-step Qwen transformer, an 8-step Lightning LoRA,
and a 4-step Rapid AIO NSFW transformer. The full IDs and defaults live in
`backend/core/config.py` and are exposed by `GET /api/config`.

No pipeline has an application-level safety checker, prompt filter, or output
classifier. Weight-level learned behavior is separate from runtime filtering.

## Editing

Qwen performs instruction editing on an aspect-ratio-preserving working copy.
Without a mask, its output is resized back to the source dimensions. With a
painted mask, the edited result is composited over the untouched upload at the
original resolution. Pixels outside the mask therefore come from the uploaded
file, not from the diffusion output.
