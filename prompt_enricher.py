# prompt_enricher.py
"""
Prompt enrichment for SDXL inpainting.

Design principle: keep the user's prompt as-is and only add a standard
negative prompt. Modern checkpoints (JuggernautXL, etc.) are trained to
produce high-quality output from clear natural-language prompts — stuffing
extra quality tokens ("8k", "sharp focus") wastes the 77-token CLIP budget
and can dilute intent.

FLUX uses T5-XXL (no token limit concerns) and guidance_scale=0, so it
doesn't benefit from prompt enrichment or negative prompts at all.
"""

NEGATIVE_DEFAULT = (
    "low quality, blurry, jpeg artifacts, bad anatomy, deformed, "
    "extra arms, extra hands, extra fingers, missing fingers, "
    "plastic skin, over-smoothed skin, uncanny, text, watermark, logo"
)


def enrich_positive(prompt: str | None) -> str:
    """Return the user's prompt as-is. No token stuffing."""
    return (prompt or "").strip()


def enrich_negative(user_negative: str | None) -> str:
    """Combine user's negative prompt with standard defaults."""
    user = (user_negative or "").strip()
    if user:
        return f"{user}, {NEGATIVE_DEFAULT}"
    return NEGATIVE_DEFAULT
