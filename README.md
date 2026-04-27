# ImageAI Studio

A high-performance, AI-powered image generation and editing studio. Built with **Next.js** and **FastAPI**, this project focuses on **efficient ML model serving, dynamic VRAM management, and optimized inference pipelines** for heavy models like FLUX and SDXL.

![License](https://img.shields.io/badge/License-MIT-blue.svg)

## 🛠 Engineering Focus & Development Process

This project was developed with a primary focus on **System Architecture and ML Inference Optimization**. 

- **System Architecture & ML Serving:** Designed a robust FastAPI backend to handle asynchronous model inference, dynamic model switching, and API routing.
- **Resource & Memory Optimization:** Implemented strict VRAM management (Soft/Hard clears, lazy model loading, and CPU offloading) to prevent Out-Of-Memory (OOM) errors and ensure stable serving on consumer-grade GPUs.
- **AI-Assisted Prototyping:** Actively utilized **Claude Code** and **Antigravity** to accelerate frontend boilerplate generation and rapid prototyping. This allowed me to concentrate my core engineering efforts on    backend architecture, memory profiling, and pipeline integration.
  

## 🚀 Core Features

- **⚡ Optimized Image Generation** — Text-to-image serving using FLUX Schnell with optimized latent processing.
- **✏️ Advanced Inpainting Pipeline** — SDXL inpainting orchestrated with ControlNet (Canny, Depth, Pose) for precise semantic editing.
- **🎭 Automated Segmentation** — Integrated MediaPipe (person), SAM (click-to-select), and SegFormer (clothing items & hair) to automate masking workflows with minimal latency.
- **🛡️ Identity Preservation Logic** — Automated facial region protection during localized edits (e.g., clothing swapping).
- **📊 Dynamic VRAM Management** — Custom memory handlers for lazy model loading, CPU offloading, and cache clearing to maximize hardware efficiency.
  

## 🏗 Architecture & Data Flow

```text
imageAI_public/
├── backend/              # FastAPI Inference Server
│   ├── main.py           # App entry, CORS, lifespan management
│   ├── core/
│   │   ├── config.py     # Hardware device routing & environment config
│   │   ├── pipeline.py   # Diffusers pipeline loading & precision tuning
│   │   └── vram.py       # Garbage collection & GPU memory state management
│   └── routers/
│       ├── generate.py   # POST /api/generate (FLUX endpoint)
│       ├── edit.py       # POST /api/edit (SDXL + ControlNet + FLUX Fill)
│       ├── kontext.py    # POST /api/kontext (FLUX Kontext mask-free edit)
│       ├── mask.py       # POST /api/mask/auto, /click (Segmentation)
│       └── system.py     # GET /api/vram, POST /api/clear (Ops endpoints)
├── frontend/             # Next.js Client
│   └── src/app/
│       ├── page.tsx      # Generation & Editing Canvas
│       └── lib/api.ts    # Typed asynchronous API client
├── prompt_enricher.py    # Negative prompt defaults for SDXL
├── segformer_masks.py    # SegFormer clothing segmentation (18-class)
└── mp_tasks_utils.py     # MediaPipe person segmentation
```

💻 Tech Stack

- **Frontend**:  Next.js 16, TypeScript, Tailwind CSS
- **Backend & Serving**:  FastAPI, Uvicorn, Python 3.11+
- **ML/AI Models**:  FLUX Schnell, FLUX Fill, FLUX Kontext, SDXL Inpaint, SDXL ControlNet
- **Computer Vision**:  SAM, MediaPipe, SegFormer (b2_clothes)
- **Infra & Optimization**:  PyTorch, Diffusers, CUDA 12.x, xformers

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
