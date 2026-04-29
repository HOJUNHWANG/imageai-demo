#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
"""
Test model downloader — auto-fetches latest versions and updates .env.local.

Sources:
  CivitAI  → uses CIVITAI_API_KEY from .env.local (required for download)
  HuggingFace → uses huggingface_hub (no key needed for public repos)

Usage:
  python scripts/download_test_models.py            # download all
  python scripts/download_test_models.py --dry-run  # show what would be downloaded
  python scripts/download_test_models.py --slot 1   # download only test_model_1

CivitAI API key:  https://civitai.com/user/account  →  API Keys
Add to .env.local:  CIVITAI_API_KEY=your_key_here
"""

import os
import sys
import re
import argparse
import requests
from pathlib import Path
from typing import Optional

# ─── Model registry ──────────────────────────────────────────────────────────
# CivitAI model page IDs are stable (civitai.com/models/{id}).
# The script fetches the latest version automatically via the CivitAI API.
# HuggingFace repos are fetched via huggingface_hub.
#
# ┌──────────────┬──────────────────────────────────────────────────────────┐
# │ Slot         │ Model                                                    │
# ├──────────────┼──────────────────────────────────────────────────────────┤
# │ test_model_1 │ RealVisXL V5  — photorealistic SDXL                      │
# │ test_model_2 │ Pony Diffusion V6 XL — general / max LoRA ecosystem      │
# │ test_model_3 │ Illustrious XL v0.1 — illustration / anime               │
# │ test_model_4 │ NoobAI XL — Illustrious-based, deeper uncensored tuning  │
# └──────────────┴──────────────────────────────────────────────────────────┘

MODELS = {
    "test_model_1": {
        "display": "RealVisXL V5.0 (Baked VAE)",
        "page": "https://civitai.com/models/139562/realvisxl-v50",
        "source": "civitai",
        "civitai_model_id": 139562,
        # Prefer the VAE-baked variant (no separate VAE needed)
        "prefer_name_fragment": "BakedVAE",
    },
    "test_model_2": {
        "display": "Pony Diffusion V6 XL",
        "page": "https://civitai.com/models/257749/pony-diffusion-v6-xl",
        "source": "civitai",
        "civitai_model_id": 257749,
        "prefer_name_fragment": None,
    },
    "test_model_3": {
        "display": "Illustrious XL v0.1",
        "page": "https://civitai.com/models/795765/illustrious-xl",
        "source": "civitai",
        "civitai_model_id": 795765,
        "prefer_name_fragment": None,
    },
    "test_model_4": {
        "display": "NoobAI XL v1.0",
        "page": "https://civitai.com/models/833294/noobai-xl-nai-xl",
        "source": "civitai",
        "civitai_model_id": 833294,
        "prefer_name_fragment": None,
    },
}

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
MODELS_DIR  = PROJECT_DIR / "models" / "test"
ENV_LOCAL   = PROJECT_DIR / ".env.local"


# ─── .env.local helpers ──────────────────────────────────────────────────────

def load_env_local() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_LOCAL.exists():
        return env
    for line in ENV_LOCAL.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
    return env


def update_env_local(key: str, value: str) -> None:
    """Set or replace a key in .env.local, preserving all comments and order."""
    content = ENV_LOCAL.read_text(encoding="utf-8") if ENV_LOCAL.exists() else ""
    pattern = re.compile(rf"^({re.escape(key)}\s*=.*)$", re.MULTILINE)
    new_line = f"{key}={value}"
    if pattern.search(content):
        content = pattern.sub(lambda _: new_line, content)
    else:
        content = content.rstrip("\n") + f"\n{new_line}\n"
    ENV_LOCAL.write_text(content, encoding="utf-8")
    print(f"  [.env.local] {key} = {value}")


# ─── CivitAI download ─────────────────────────────────────────────────────────

def civitai_get_latest_file(model_id: int, api_key: str, prefer_fragment: Optional[str]) -> tuple[str, str]:
    """
    Returns (download_url, filename) for the latest primary .safetensors file.
    Prefers a variant whose name contains prefer_fragment (e.g. 'BakedVAE') when set.
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(
        f"https://civitai.com/api/v1/models/{model_id}",
        headers=headers,
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()

    versions = data.get("modelVersions", [])
    if not versions:
        raise RuntimeError(f"No versions found for model {model_id}")

    latest = versions[0]  # CivitAI returns newest first
    files = latest.get("files", [])

    # Filter to .safetensors only
    sf_files = [f for f in files if f.get("name", "").endswith(".safetensors")]
    if not sf_files:
        raise RuntimeError(f"No .safetensors files in model {model_id} latest version")

    # Pick preferred variant if specified
    chosen = sf_files[0]
    if prefer_fragment:
        for f in sf_files:
            if prefer_fragment.lower() in f["name"].lower():
                chosen = f
                break

    url = chosen["downloadUrl"]
    name = chosen["name"]
    return url, name


def download_file(url: str, dest: Path, api_key: Optional[str] = None) -> None:
    """Stream-download with a simple progress bar."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
                        mb_done = downloaded / 1e6
                        mb_total = total / 1e6
                        print(f"\r  [{bar}] {pct:5.1f}%  {mb_done:.0f}/{mb_total:.0f} MB", end="", flush=True)
        print()  # newline after progress bar


# ─── HuggingFace download ─────────────────────────────────────────────────────

def download_hf(repo_id: str, filename: str, dest_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  [ERROR] huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    print(f"  Downloading from HuggingFace: {repo_id}/{filename}")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )
    return Path(path)


# ─── Main ─────────────────────────────────────────────────────────────────────

def process_slot(slot_key: str, cfg: dict, api_key: Optional[str], dry_run: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"  {slot_key}  →  {cfg['display']}")
    print(f"  Page: {cfg['page']}")

    source = cfg["source"]

    if source == "civitai":
        if not api_key:
            print("  [SKIP] CIVITAI_API_KEY not set in .env.local")
            print(f"         Get yours at: https://civitai.com/user/account")
            print(f"         Then add to .env.local:  CIVITAI_API_KEY=your_key")
            return False

        try:
            url, filename = civitai_get_latest_file(
                cfg["civitai_model_id"], api_key, cfg.get("prefer_name_fragment")
            )
        except Exception as e:
            print(f"  [ERROR] CivitAI API call failed: {e}")
            return False

        dest = MODELS_DIR / filename
        print(f"  File: {filename}")

        if dest.exists():
            print(f"  [SKIP] Already downloaded ({dest.stat().st_size / 1e9:.2f} GB)")
        elif dry_run:
            print(f"  [DRY-RUN] Would download to: {dest}")
        else:
            print(f"  Downloading...")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp")
            try:
                download_file(url, tmp, api_key)
                tmp.rename(dest)
                print(f"  Saved: {dest}")
            except Exception as e:
                if tmp.exists():
                    tmp.unlink()
                print(f"  [ERROR] Download failed: {e}")
                return False

        if not dry_run:
            update_env_local(slot_key.upper(), str(dest).replace("/", "\\"))
        return True

    elif source == "huggingface":
        repo_id = cfg["repo_id"]
        filename = cfg["filename"]
        dest = MODELS_DIR / filename

        if dest.exists():
            print(f"  [SKIP] Already downloaded ({dest.stat().st_size / 1e9:.2f} GB)")
        elif dry_run:
            print(f"  [DRY-RUN] Would download {repo_id}/{filename}")
        else:
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                download_hf(repo_id, filename, MODELS_DIR)
                print(f"  Saved: {dest}")
            except Exception as e:
                print(f"  [ERROR] HuggingFace download failed: {e}")
                return False

        if not dry_run:
            update_env_local(slot_key.upper(), str(dest).replace("/", "\\"))
        return True

    else:
        print(f"  [ERROR] Unknown source: {source}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download test models and update .env.local")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--slot", type=int, choices=[1, 2, 3, 4], help="Download only a specific slot (1-4)")
    args = parser.parse_args()

    env = load_env_local()
    api_key = env.get("CIVITAI_API_KEY") or os.getenv("CIVITAI_API_KEY")

    if not api_key:
        print("\n⚠️  CIVITAI_API_KEY not found.")
        print("   CivitAI models require an API key for download.")
        print("   1. Go to https://civitai.com/user/account → API Keys")
        print("   2. Add to .env.local:  CIVITAI_API_KEY=your_key_here\n")

    print(f"\nModels dir : {MODELS_DIR}")
    print(f"Dry run    : {args.dry_run}")

    results = {}
    target_models = MODELS
    if args.slot:
        key = f"test_model_{args.slot}"
        target_models = {key: MODELS[key]}

    for slot_key, cfg in target_models.items():
        ok = process_slot(slot_key, cfg, api_key, args.dry_run)
        results[slot_key] = ok

    print(f"\n{'='*60}")
    print("Summary:")
    for k, ok in results.items():
        status = "✓ OK" if ok else "✗ SKIPPED/FAILED"
        print(f"  {k}: {status}")

    if not args.dry_run and any(results.values()):
        print("\n.env.local updated — restart the backend to load new models.")


if __name__ == "__main__":
    main()
