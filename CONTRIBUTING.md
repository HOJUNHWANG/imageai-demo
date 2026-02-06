# Contributing

Thanks for your interest!

This repository is a **public/portfolio-friendly demo**. Please keep PRs focused on:
- Documentation improvements
- Bug fixes
- Small UX enhancements
- Safer defaults (public demo guardrails)

## Development setup
- Python 3.11
- Create a venv and install requirements:

```bash
python -m venv venv311
.\venv311\Scripts\activate
pip install -r requirements.txt
```

Optional components:
- SAM (Segment Anything):
  ```bash
  pip install git+https://github.com/facebookresearch/segment-anything.git
  ```

## What not to include
- Model weights (e.g., `.safetensors`, `.pth`), large binaries
- Personal images / private datasets
- Secrets (`.env`, tokens)

## Style
- Keep changes minimal and readable
- Prefer additive changes over large refactors
