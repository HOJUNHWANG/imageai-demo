# ImageAI: High-Performance Local Generative AI System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio&logoColor=white)

> **A robust local AI application engineering two distinct diffusion models (FLUX.1-schnell & SDXL) into a unified, memory-efficient workflow.**
>
> *Developed to mitigate the "VRAM bottleneck" on consumer hardware while delivering state-of-the-art generation and editing capabilities.*

---

## 🏗️ Engineering Highlights

This project demonstrates **System Engineering** and **AI Pipeline Architecture** capabilities by solving key challenges in local AI deployment.

### 1. Hybrid Model Orchestration (AI Engineer Focus)
Instead of relying on a single monolithic model, this system orchestrates two specialized models to maximize quality and performance:
- **Generation Layer (FLUX.1-schnell)**: Utilized for its superior prompt adherence and 4-step distilled inference speed.
- **Editing Layer (Stable Diffusion XL Inpaint)**: Integrated for its precise masking capabilities and high-resolution denoising strength.
- **Result:** A seamless workflow where users generate high-fidelity bases in FLUX and perform pixel-perfect edits in SDXL without context switching.

### 2. Resource-Constrained Optimization (Backend Dev Focus)
Running 30GB+ of model weights on a single 12GB Consumer GPU required aggressive resource management strategies:
- **Dynamic Pipeline Offloading**: Implemented a custom context manager that monitors user intent (Tab Switching) to automatically move idle models to CPU/RAM, preventing Out-Of-Memory (OOM) crashes.
- **Precision Engineering**: Enforced `torch.bfloat16` inference (where supported) to reduce memory footprint by 50% while maintaining numerical stability.
- **Garbage Collection Strategy**: Developed `soft_clear()` (cache only) and `hard_clear()` (full unload) protocols to recover fragmented VRAM during long sessions.

### 3. Robust Dependency Resolution
- **"DLL Hell" Mitigation**: Solved critical version conflicts between `huggingface_hub`, `transformers`, and `diffusers` through automated environment patching hotfixes.
- **Environment Isolation**: Strictly scoped `requirements.txt` to tested versions to ensure reproducibility.

---

## 🚀 Key Features

- **Text-to-Image**: Instant 4-step generation using FLUX.1 models.
- **Smart Inpainting**:
    - **Manual**: Precision masking with **Segment Anything Model (SAM)** integration.
    - **Auto**: Semantic segmentation using **MediaPipe** (automatically detects Face, Clothes, Background).
- **Non-Blocking UI**: Asynchronous loading states for smooth user experience during massive model swaps.

---

## 📦 Installation

### Prerequisites
- **Python 3.11** (Strict requirement for dependency compatibility)
- **NVIDIA GPU** (RTX 3060 12GB or higher recommended)

### Setup Schema
```bash
# 1. Clone & Environment
git clone https://github.com/HOJUNHWANG/imageai-demo.git
python -m venv venv311
.\venv311\Scripts\activate

# 2. Install Dependencies
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git

# 3. Authenticate (Required for FLUX)
huggingface-cli login
```

---

## 🖥️ Usage Guide

1. **Launch System**: `python app.py`
2. **Dashboard**: Navigate to `http://localhost:7860`
3. **Operations**:
   - **Gen Tab**: Prompt -> Generate (FLUX loads automatically).
   - **Edit Tab**: Upload/Send Image -> Select Mask -> Apply (SDXL loads automatically).

---

## 📝 License
MIT License.
