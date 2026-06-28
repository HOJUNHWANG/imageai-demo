# Morrow — Local Image Studio

A local image-generation and instruction-editing studio tuned for an RTX 3080 Ti
(12 GB VRAM). Generate and Edit each expose three real model profiles.

## Model profiles

| Workflow | Profile | Weights | Runtime |
| --- | --- | --- | --- |
| Generate | Quality | `kpsss34/FHDR_Uncensored` | FLUX, 40 steps, 1024 px |
| Generate | Balanced | `Bl4ckSpaces/z-image-turbo-nsfw-nf4` | pre-quantized Z-Image, 8 steps, 1024 px |
| Generate | Fast | `KaraKaraWitch/Z-Image-Turbo-TE-Heretic` | Z-Image + abliterated encoder, 8 steps, 896 px |
| Edit | Quality | `Qwen/Qwen-Image-Edit-2511` | original transformer, 40 steps, 1024 px |
| Edit | Balanced | Qwen 2511 + `lightx2v/...Lightning` | distilled LoRA, 8 steps, 896 px |
| Edit | Fast | `prithivMLmods/...Rapid-AIO-V23` | NSFW rapid transformer, 4 steps, 768 px |

The application has no prompt filter, output filter, or Diffusers safety-checker
component. That does not guarantee that every base model is free of learned
refusals. The profile metadata names whether a checkpoint is NSFW-tuned,
uncensored, abliterated, or unchanged.

## RTX 3080 Ti optimizations

- Exactly one pipeline stays resident. Repeating the same profile is warm;
  changing workflow or profile unloads it before loading the next one.
- Transformer and large text encoder are NF4-quantized when necessary. This is
  important because Qwen's vision-language encoder and FLUX's T5 encoder do not
  fit comfortably in 12 GB at BF16.
- On this RTX 3080 Ti, the pinned `xformers==0.0.29.post2` backend benchmarked
  21–30% faster than native SDPA for long image-attention sequences. Startup
  installs it only for the matching Torch 2.6/CUDA 12.4 ABI and otherwise falls
  back to native SDPA.
- CPU model offload, VAE tiling/slicing, TF32, bounded pixel budgets, and fixed
  distilled step counts reduce VRAM pressure and unnecessary work.
- Every result reports model-load, inference, and post-processing time
  separately. A zero load time means the active model was reused.
- Live progress distinguishes first-use download, model loading, input
  preparation, denoising, compositing, and PNG encoding. Denoising percentage
  and ETA come from real scheduler callbacks; unknown download time remains
  visibly indeterminate instead of showing a fabricated percentage.
- Masked edits are composited at the upload's original resolution; pixels
  outside the mask come directly from the source.

## First run

1. Copy `.env.local.example` to `.env.local`.
2. Create a Hugging Face read token and put it in `HF_TOKEN`.
3. While logged in, accept the access terms for `kpsss34/FHDR_Uncensored` and
   `Bl4ckSpaces/z-image-turbo-nsfw-nf4`.
4. Run `start.bat`.

Models download lazily. The first request for a profile can therefore take much
longer than later requests and the six repositories require substantial disk
space. The frontend is at `http://localhost:3000`; the API is at
`http://127.0.0.1:8000`.

Downloading every repository before opening the UI is intentionally not the
default: their complete Hub repositories total roughly 281 GB, two generation
repos require prior access approval, and the Lightning repository contains many
unused variants. Lazy component loading downloads only files required by the
selected profile and lets the UI report model-loading state.

## API

- `POST /api/generate` with `profile: quality | balanced | fast`
- `POST /api/edit` multipart form with the same `profile` field
- `GET /api/status`
- `GET /api/config`
- `POST /api/cancel`
- `POST /api/unload`

Application code is MIT licensed. Model weights retain their own licenses and
usage terms.
