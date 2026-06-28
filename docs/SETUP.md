# Setup

1. Copy `.env.local.example` to `.env.local`.
2. Add a Hugging Face read token to `HF_TOKEN`.
3. Accept the model terms for the gated Quality and Balanced generation repos
   listed in `README.md`.
4. Run `start.bat` and open `http://localhost:3000`.

Each profile downloads lazily. First-use load time includes network and cache
work; repeat requests on the same profile reuse the loaded pipeline. Switching
profile or workflow intentionally keeps only one pipeline resident to stay
within 12 GB VRAM and reasonable system RAM.

Run `venv311\Scripts\python.exe scripts\doctor.py` to verify dependencies and
CUDA without loading weights. Set `LOCAL_FILES_ONLY=1` after every required
profile is cached if the machine should run fully offline.
