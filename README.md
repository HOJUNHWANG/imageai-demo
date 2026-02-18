# ImageAI: Advanced Generative Image Processing System

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Diffusers](https://img.shields.io/badge/HuggingFace-Diffusers-FFD21E?logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)

> **A high-performance local AI application integrating Stable Diffusion XL (Inpainting) and FLUX.1-schnell (Generation) into a unified workflow.** Designed for creative professionals, featuring smart VRAM management and advanced masking tools.

---

## 🚀 Key Features

### 1. Hybrid Generative Pipeline
Seamlessly switch between two state-of-the-art models within a single interface:
- **Text-to-Image (FLUX.1-schnell)**: Ultra-fast 4-step generation for high-fidelity base images.
- **Inpainting (SDXL 1.0)**: Precision editing using mask-based diffusion for realistic object modification.

### 2. Smart VRAM Optimization (Consumer GPU Friendly)
Engineered to run heavy transformer models on consumer GPUs (e.g., RTX 3080 Ti, 12GB VRAM):
- **Dynamic Model Offloading**: Automatically unloads idle pipelines (e.g., checks if user is Editing vs Generating) to free up VRAM.
- **Memory-Efficient Precision**: Utilizes `bfloat16` (if available) or `float16` to halve memory footprint without sacrificing quality.
- **Aggressive Garbage Collection**: Implements a custom `hard_clear_vram()` mechanism to recover memory from fragmented CUDA caches during heavy workloads.

### 3. Advanced Masking System
Combines multiple AI vision models for precise selection:
- **SAM (Segment Anything Model)**: Click-to-segment functionality for manual, pixel-perfect masks.
- **MediaPipe Integration**: Semantic segmentation to automatically mask specific body parts (Head, Face, Clothes, Background) with one click.

### 4. Robust Engineering
- **Dependency Isolation**: Resolves complex version conflicts (e.g., `transformers` vs `huggingface_hub`) via automated environment patches.
- **Port Conflict Resolution**: Auto-detects available ports (7860-7870) to ensure reliable startup.

---

## 🛠️ Technical Architecture

```mermaid
graph TD
    User[User Interface (Gradio)] -->|Select Tab| ModeManager{Mode Switcher}
    
    ModeManager -->|Generate Tab| FLUX[FLUX.1-schnell Pipeline]
    ModeManager -->|Edit Tab| SDXL[SDXL Inpaint Pipeline]
    
    subgraph "VRAM Optimization Layer"
        FLUX -.->|Unload| CPU[System RAM]
        SDXL -.->|Unload| CPU
        CPU -.->|Load on Demand| GPU[GPU VRAM (12GB)]
    end
    
    subgraph "Masking Engine"
        SDXL --> SAM[Segment Anything]
        SDXL --> MP[MediaPipe Semantic]
    end
```

---

## 📦 Installation

### Prerequisites
- Python 3.11
- NVIDIA GPU (Recommended: 12GB+ VRAM)
- CUDA Toolkit 11.8+

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/HOJUNHWANG/imageai-demo.git
   cd imageai-demo
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv311
   .\venv311\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   # Install Segment Anything (SAM) manually
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

4. **Authentication (For FLUX Model)**
   This project uses `FLUX.1-schnell`, a gated model.
   ```bash
   huggingface-cli login
   # Enter your Hugging Face Access Token with 'Write' permissions
   ```

---

## 🖥️ Usage

1. **Start the Application**
   ```bash
   python app.py
   ```
2. **Open Browser**
   Access the UI at `http://127.0.0.1:7860`.

3. **Workflow Example**
   - **Step 1 (Generate)**: Go to "Text to Image" tab. Enter "cyberpunk street", click Generate.
   - **Step 2 (Transfer)**: Click "Send to Image Editor".
   - **Step 3 (Edit)**: In "Image Editing" tab, use "Auto Mask" to select the sky. Enter "starry night sky" and Apply.

---

## 📝 License
MIT License. See `LICENSE` for details.

---

*This project was developed to demonstrate advanced integration of large-scale generative models in a local environment.*
