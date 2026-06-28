"""Runtime configuration and the six model profiles."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    model_id: str
    family: str
    description: str
    steps: int
    guidance: float
    true_cfg: float = 1.0
    long_side: int = 1024
    max_pixels: int = 1024 * 1024
    transformer_id: str | None = None
    transformer_subfolder: str | None = "transformer"
    lora_id: str | None = None
    lora_weight: str | None = None
    prequantized: bool = False
    gated: bool = False
    content_tuning: str = "none"

    def public(self) -> dict:
        data = asdict(self)
        data.pop("transformer_subfolder")
        return data


GENERATE_PROFILES = {
    "quality": ModelProfile(
        id="quality",
        label="Quality",
        model_id=os.getenv("GENERATE_QUALITY_MODEL", "kpsss34/FHDR_Uncensored"),
        family="flux",
        description="Highest detail and prompt fidelity; slow 40-step FLUX render.",
        steps=40,
        guidance=4.0,
        gated=True,
        content_tuning="uncensored",
    ),
    "balanced": ModelProfile(
        id="balanced",
        label="Balanced",
        model_id=os.getenv("GENERATE_BALANCED_MODEL", "Bl4ckSpaces/z-image-turbo-nsfw-nf4"),
        family="zimage",
        description="NSFW-tuned, pre-quantized Z-Image with a good speed/detail tradeoff.",
        steps=8,
        guidance=0.0,
        prequantized=True,
        gated=True,
        content_tuning="nsfw",
    ),
    "fast": ModelProfile(
        id="fast",
        label="Fast",
        model_id=os.getenv("GENERATE_FAST_MODEL", "KaraKaraWitch/Z-Image-Turbo-TE-Heretic"),
        family="zimage",
        description="Fast 8-step Z-Image with an abliterated text encoder for fewer refusals.",
        steps=8,
        guidance=0.0,
        long_side=896,
        max_pixels=896 * 896,
        content_tuning="abliterated text encoder",
    ),
}

EDIT_BASE_MODEL = os.getenv("EDIT_BASE_MODEL", "Qwen/Qwen-Image-Edit-2511")
EDIT_PROFILES = {
    "quality": ModelProfile(
        id="quality",
        label="Quality",
        model_id=EDIT_BASE_MODEL,
        family="qwen_edit",
        description="Full 40-step Qwen 2511 edit for the strongest source consistency.",
        steps=40,
        guidance=1.0,
        true_cfg=4.0,
        content_tuning="base weights",
    ),
    "balanced": ModelProfile(
        id="balanced",
        label="Balanced",
        model_id=EDIT_BASE_MODEL,
        family="qwen_edit",
        description="Official 8-step Lightning adapter; much faster with modest detail loss.",
        steps=8,
        guidance=1.0,
        true_cfg=1.0,
        long_side=896,
        max_pixels=896 * 896,
        lora_id=os.getenv("EDIT_BALANCED_LORA", "lightx2v/Qwen-Image-Edit-2511-Lightning"),
        lora_weight="Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors",
        content_tuning="distilled LoRA",
    ),
    "fast": ModelProfile(
        id="fast",
        label="Fast",
        model_id=EDIT_BASE_MODEL,
        family="qwen_edit",
        description="Rapid AIO NSFW v23 transformer distilled for four-step edits.",
        steps=4,
        guidance=1.0,
        true_cfg=1.0,
        long_side=768,
        max_pixels=768 * 768,
        transformer_id=os.getenv("EDIT_FAST_TRANSFORMER", "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23"),
        transformer_subfolder=None,
        content_tuning="NSFW rapid transformer",
    ),
}

DEFAULT_PROFILE = os.getenv("DEFAULT_PROFILE", "balanced")
ENABLE_4BIT = os.getenv("ENABLE_4BIT", "1") == "1"
QUANTIZE_TEXT_ENCODERS = os.getenv("QUANTIZE_TEXT_ENCODERS", "1") == "1"
CPU_OFFLOAD = os.getenv("CPU_OFFLOAD", "1") == "1"
ATTENTION_BACKEND = os.getenv("ATTENTION_BACKEND", "xformers")
LOCAL_FILES_ONLY = os.getenv("LOCAL_FILES_ONLY", "0") == "1"
HF_TOKEN = os.getenv("HF_TOKEN") or None

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "32"))
MAX_LONG_SIDE = int(os.getenv("MAX_LONG_SIDE", "1280"))
MAX_PIXELS = int(os.getenv("MAX_PIXELS", str(1024 * 1024)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

if DEVICE == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def get_profile(kind: str, profile_id: str) -> ModelProfile:
    profiles = GENERATE_PROFILES if kind == "generate" else EDIT_PROFILES
    try:
        return profiles[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown {kind} profile: {profile_id}") from exc
