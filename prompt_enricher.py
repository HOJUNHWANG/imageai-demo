# prompt_enricher.py
"""
Prompt enrichment system with edit-type-specific presets.
Each preset focuses tokens on the user's intent instead of flooding with generic quality tokens.
"""

# ─── Shared basic quality (kept minimal to not dilute intent) ─────
QUALITY_CORE = ["photorealistic", "high detail", "realistic lighting", "sharp focus", "8k"]

NEGATIVE_DEFAULT = (
    "low quality, blurry, jpeg artifacts, bad anatomy, deformed, extra arms, "
    "extra hands, extra fingers, missing fingers, plastic skin, over-smoothed skin, "
    "uncanny, text, watermark, logo"
)

# ─── Korean → English color map ─────
KOR_COLOR = {
    "검은": "black", "검정": "black", "블랙": "black",
    "하얀": "white", "흰": "white", "화이트": "white",
    "빨간": "red", "빨강": "red", "레드": "red",
    "파란": "blue", "파랑": "blue", "블루": "blue",
    "초록": "green", "녹색": "green", "그린": "green",
    "회색": "gray", "그레이": "gray",
    "갈색": "brown", "브라운": "brown",
    "베이지": "beige", "아이보리": "ivory",
    "노란": "yellow", "노랑": "yellow", "옐로우": "yellow",
    "보라": "purple", "퍼플": "purple",
    "분홍": "pink", "핑크": "pink",
    "주황": "orange", "오렌지": "orange",
}


def _detect_color(text: str) -> str | None:
    """Detect color from prompt (Korean or English)."""
    t = text.lower()
    for kor, eng in KOR_COLOR.items():
        if kor in t:
            return eng
    for c in ["black", "white", "red", "blue", "green", "gray", "brown",
              "beige", "ivory", "yellow", "purple", "pink", "orange",
              "navy", "burgundy", "cream", "khaki", "olive"]:
        if c in t:
            return c
    return None


# ═══════════════════════════════════════════════════════════
# ENRICHER PRESETS — each optimized for a specific edit type
# ═══════════════════════════════════════════════════════════

PRESETS = {
    "color_change": {
        "label": "🎨 Color Change",
        "description": "Optimized for changing colors of clothing/objects",
        "tokens": [
            "exact color match", "vivid color", "consistent coloring",
            "same fabric texture", "uniform color distribution",
        ],
        "negative_extra": "wrong color, color bleeding, patchy color, inconsistent tone",
        "strength": 0.80,
        "guidance": 8.0,
        "steps": 28,
    },
    "cloth_change": {
        "label": "👕 Cloth/Garment Change",
        "description": "Change clothing type (shirt → jacket, etc.)",
        "tokens": [
            "detailed fabric texture", "realistic clothing folds",
            "natural draping", "consistent lighting on fabric",
            "clean garment edges", "proper fit",
        ],
        "negative_extra": "deformed clothing, floating fabric, unrealistic folds",
        "strength": 0.75,
        "guidance": 7.5,
        "steps": 30,
    },
    "hair_change": {
        "label": "💇 Hair Change",
        "description": "Change hairstyle or hair color",
        "tokens": [
            "realistic hair strands", "natural hair texture",
            "volumetric hair", "hair highlights and shadows",
            "natural hairline", "individual hair strands visible",
        ],
        "negative_extra": "wig-like, plastic hair, unnatural hairline, bald spots",
        "strength": 0.72,
        "guidance": 7.5,
        "steps": 28,
    },
    "background_change": {
        "label": "🌄 Background Change",
        "description": "Replace or modify background",
        "tokens": [
            "detailed background", "realistic environment",
            "natural depth of field", "atmospheric perspective",
            "consistent lighting", "realistic shadows",
        ],
        "negative_extra": "floating objects, inconsistent shadows, obvious compositing",
        "strength": 0.90,
        "guidance": 8.5,
        "steps": 28,
    },
    "style_mood": {
        "label": "✨ Style / Mood",
        "description": "Change overall style or atmosphere",
        "tokens": [
            "artistic style", "cohesive aesthetic",
            "color grading", "atmospheric lighting",
            "cinematic composition",
        ],
        "negative_extra": "inconsistent style, jarring colors",
        "strength": 0.65,
        "guidance": 7.0,
        "steps": 28,
    },
    "face_retouch": {
        "label": "👤 Face Retouch",
        "description": "Subtle face retouching (expression, makeup, etc.)",
        "tokens": [
            "natural skin texture", "realistic facial features",
            "natural expression", "subsurface scattering on skin",
            "natural skin pores", "realistic eye detail",
        ],
        "negative_extra": "uncanny valley, plastic face, asymmetric features, cross-eyed",
        "strength": 0.50,
        "guidance": 6.0,
        "steps": 24,
    },
    "general": {
        "label": "⚙️ General",
        "description": "General purpose editing (legacy mode)",
        "tokens": [
            "natural skin texture", "consistent skin tone",
            "clean edges", "consistent shading",
        ],
        "negative_extra": "",
        "strength": 0.55,
        "guidance": 7.0,
        "steps": 28,
    },
}

GEN_PRESETS = {
    "portrait_closeup": {
        "label": "👤 Close-up Portrait",
        "description": "Tight framing on the face and shoulders",
        "tokens": [
            "extreme close-up portrait", "85mm lens", "f/1.4", "shallow depth of field",
            "detailed facial features", "catchlights in eyes", "sharp focus on eyes",
        ]
    },
    "medium_shot": {
        "label": "📸 Medium Shot (상반신)",
        "description": "Waist-up framing, good for fashion tops",
        "tokens": [
            "medium shot", "waist up portrait", "50mm lens", "f/2.8",
            "balanced framing", "environmental context",
        ]
    },
    "full_body": {
        "label": "🧍 Full Body (전신)",
        "description": "Full body visible from head to toe",
        "tokens": [
            "wide shot", "full body portrait", "standing pose", "head to toe",
            "35mm lens", "f/4.0", "subject fully in frame",
        ]
    },
    "pov_selfie": {
        "label": "🤳 Selfie POV",
        "description": "Looks like an authentic smartphone selfie",
        "tokens": [
            "selfie perspective", "front facing camera", "smartphone photo",
            "arm slightly visible", "casual pose", "authentic amateur photography",
        ]
    },
    "high_angle": {
        "label": "🚁 High Angle (버드아이)",
        "description": "Looking down at the subject",
        "tokens": [
            "high angle shot", "looking down at subject", "bird's eye view",
            "camera elevated", "dynamic perspective",
        ]
    },
    "low_angle": {
        "label": "🐜 Low Angle (로우 앵글)",
        "description": "Looking up at the subject, making them appear taller",
        "tokens": [
            "low angle shot", "looking up at subject", "worm's eye view",
            "heroic perspective", "camera looking up",
        ]
    },
    "cinematic_light": {
        "label": "🎬 Cinematic Lighting",
        "description": "Dramatic mood with rim lights and deep shadows",
        "tokens": [
            "cinematic lighting", "dramatic shadows", "rim lighting", "neon accents",
            "teal and orange color grading", "high contrast", "moody atmosphere",
        ]
    },
    "studio_light": {
        "label": "💡 Studio Lighting",
        "description": "Clean, even, bright lighting for e-commerce/fashion",
        "tokens": [
            "clean studio lighting", "softbox", "bright and airy", "even illumination",
            "fashion editorial lighting", "white backdrop", "no harsh shadows",
        ]
    },
    "general": {
        "label": "⚙️ Default Mode",
        "description": "Standard photorealistic generation",
        "tokens": [
            "professional photography", "natural lighting", "highly detailed",
        ]
    }
}

def get_preset_list() -> list[dict]:
    """Return preset info for frontend dropdown."""
    return [
        {
            "id": k,
            "label": v["label"],
            "description": v["description"],
            "strength": v["strength"],
            "guidance": v["guidance"],
            "steps": v["steps"],
        }
        for k, v in PRESETS.items()
    ]

def get_gen_preset_list() -> list[dict]:
    """Return preset info for Generation tab dropdown."""
    return [
        {
            "id": k,
            "label": v["label"],
            "description": v["description"],
        }
        for k, v in GEN_PRESETS.items()
    ]

def enrich_positive(text: str | None, preset: str = "general") -> tuple[str, str]:
    """Enrich prompt using selected preset.

    Key design: user's intent is placed FIRST and repeated for emphasis.
    Preset tokens are added after to support (not override) the intent.
    """
    raw = (text or "").strip()

    # Get preset config (fallback to general)
    config = PRESETS.get(preset, PRESETS["general"])
    preset_tokens = config["tokens"]

    # Detect color for color-specific presets
    color = _detect_color(raw) if raw else None

    # Build token list — user intent first, then preset tokens, then minimal quality
    tokens = []

    # For color change preset: emphasize the color strongly
    if preset == "color_change" and color:
        tokens.extend([f"{color} colored", f"vivid {color}", f"bright {color} fabric"])

    tokens.extend(preset_tokens)
    tokens.extend(QUALITY_CORE)

    expanded = ", ".join(tokens)

    # Combine: user intent FIRST (most weight in CLIP), then tokens
    if raw:
        combined = f"{raw}, {expanded}"
    else:
        combined = expanded

    # CLIP 77 token limit bypass (BREAK)
    token_list = combined.split(", ")
    if len(token_list) > 75:
        mid = len(token_list) // 2
        part1 = ", ".join(token_list[:mid])
        part2 = ", ".join(token_list[mid:])
        combined = f"{part1} BREAK {part2}"

    info = f"preset={preset}, color={color}"
    return combined, info


def enrich_negative(text: str | None, preset: str = "general") -> str:
    """Negative prompt enrichment with preset-specific additions."""
    t = (text or "").strip()
    config = PRESETS.get(preset, PRESETS["general"])
    extra = config.get("negative_extra", "")

    base_tokens = [tok.strip() for tok in NEGATIVE_DEFAULT.split(",") if tok.strip()]
    if extra:
        base_tokens.extend([tok.strip() for tok in extra.split(",") if tok.strip()])
    if t:
        base_tokens.extend([tok.strip() for tok in t.split(",") if tok.strip()])
    return ", ".join(base_tokens)

def enrich_flux(prompt: str, preset: str = "general") -> tuple[str, str]:
    """Enrich prompt for FLUX using selected generation preset."""
    raw = (prompt or "").strip()
    
    config = GEN_PRESETS.get(preset, GEN_PRESETS["general"])
    preset_tokens = config.get("tokens", [])
    
    tokens = []
    if raw:
        tokens.append(raw)
    tokens.extend(preset_tokens)
    
    # Optional defaults for generating people/faces generally if no specific style requested
    if preset == "general":
        tokens.extend([
            "natural lighting", "highly detailed", "8k resolution"
        ])
        
    enriched = ", ".join(tokens)
    
    # FLUX negative prompt is usually ignored or minimal
    negative = (
        "cartoon, anime, 3d render, CGI, plastic skin, "
        "bad anatomy, deformed, extra fingers, text, watermark, blurry"
    )
    
    return enriched, negative