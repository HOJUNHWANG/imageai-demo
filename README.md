# ImageAI Studio

AI-powered image generation and editing studio built with **Next.js** frontend and **FastAPI** backend, powered by **FLUX** (text-to-image) and **SDXL** (inpainting).

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Features

- **🎨 Image Generation** — Text-to-image using FLUX Schnell
- **✏️ Image Editing** — AI inpainting with SDXL + ControlNet (Canny, Depth, Pose)
- **🎭 Auto Masking** — MediaPipe (person), SAM (click-to-select), and SegFormer (clothing items & hair)
- **🛡️ Face Protection** — Automatically preserve facial identity during clothing edits
- **✨ Prompt Enrichment** — Auto-expand simple prompts using specialized Generation and Edit presets
- **📊 VRAM Management** — Soft/hard clear, lazy model loading, and CPU offloading

## Architecture

```
imageAI_public/
├── backend/              # FastAPI server
│   ├── main.py           # App entry, CORS, lifespan
│   ├── core/
│   │   ├── config.py     # Device, paths, flags
│   │   ├── pipeline.py   # Model loading & switching
│   │   └── vram.py       # GPU memory management
│   └── routers/
│       ├── generate.py   # POST /api/generate
│       ├── edit.py       # POST /api/edit
│       ├── mask.py       # POST /api/mask/auto
│       └── system.py     # GET /api/vram, POST /api/clear
├── frontend/             # Next.js app
│   └── src/app/
│       ├── page.tsx      # Generate + Edit pages
│       ├── globals.css   # Dark theme design system
│       └── lib/api.ts    # Typed API client
├── prompt_enricher.py    # Prompt expansion
├── sam_utils.py          # SAM segmentation
├── mp_tasks_utils.py     # MediaPipe masking
└── start.bat             # One-click launcher
```

## Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM (RTX 3080 Ti recommended)
- **Python**: 3.11+
- **Node.js**: 20+ (for frontend)
- **CUDA**: 12.x

## Quick Start

### Option 1: One-click
```bat
start.bat
```

### Option 2: Manual
```bash
# Terminal 1: Backend
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Terminal 2: Frontend
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | FLUX text-to-image |
| `POST` | `/api/edit` | SDXL inpainting |
| `POST` | `/api/mask/auto` | Auto-mask generation |
| `POST` | `/api/upload` | Upload working image |
| `GET` | `/api/vram` | VRAM status |
| `POST` | `/api/clear/soft` | Cache clear |
| `POST` | `/api/clear/hard` | Full model unload |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 16, TypeScript, Tailwind CSS |
| Backend | FastAPI, Uvicorn |
| AI Models | FLUX Schnell, SDXL Inpaint, SDXL ControlNet |
| Segmentation | SAM, MediaPipe, SegFormer (b2_clothes) |
| GPU | PyTorch, Diffusers, CUDA, xformers |

## License

MIT — see [LICENSE](LICENSE)
