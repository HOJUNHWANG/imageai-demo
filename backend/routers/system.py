"""
System router: VRAM monitoring, cleanup, and inference progress tracking.
"""
import time
from fastapi import APIRouter
from ..core.vram import get_vram_info, get_vram_text, soft_clear_vram
from ..core.pipeline import hard_clear_vram, unload_aux_pipelines

router = APIRouter()

# ─── Global progress state ───
PROGRESS = {
    "active": False,
    "task": "",          # "generate" or "edit"
    "step": 0,
    "total": 0,
    "status": "idle",    # "loading_model", "running", "encoding", "done", "error"
    "message": "",
    "started_at": 0,
    "elapsed": 0,
}


def reset_progress():
    PROGRESS["active"] = False
    PROGRESS["step"] = 0
    PROGRESS["total"] = 0
    PROGRESS["status"] = "idle"
    PROGRESS["message"] = ""
    PROGRESS["elapsed"] = 0
    PROGRESS["started_at"] = 0


def set_progress(task: str, status: str, message: str, step: int = 0, total: int = 0):
    PROGRESS["active"] = status not in ("idle", "done", "error")
    PROGRESS["task"] = task
    PROGRESS["step"] = step
    PROGRESS["total"] = total
    PROGRESS["status"] = status
    PROGRESS["message"] = message
    if status == "loading_model":
        PROGRESS["started_at"] = time.time()
    if PROGRESS["started_at"] > 0:
        PROGRESS["elapsed"] = round(time.time() - PROGRESS["started_at"], 1)

CANCEL_FLAG = False


def request_cancel():
    global CANCEL_FLAG
    CANCEL_FLAG = True
    print("[CANCEL] Cancel requested by user")


def clear_cancel():
    global CANCEL_FLAG
    CANCEL_FLAG = False


def check_cancel(task: str = "", pipe=None):
    """Check if cancel was requested. Call between pipeline phases."""
    if CANCEL_FLAG:
        if task:
            set_progress(task, "error", "Cancelled by user")
        if pipe and hasattr(pipe, "_interrupt"):
            pipe._interrupt = True
        print(f"[CANCEL] Cancel detected in {task or 'unknown'}")
        raise RuntimeError("CANCELLED_BY_USER")


def make_step_callback(task: str, total_steps: int):
    """Create a diffusers pipeline callback that updates progress and checks cancel."""
    def callback(pipe, step, timestep, kwargs):
        if CANCEL_FLAG:
            print(f"[CANCEL] Cancel detected at step {step + 1}/{total_steps}")
            set_progress(task, "error", "Cancelled by user")
            if hasattr(pipe, "_interrupt"):
                pipe._interrupt = True
            raise RuntimeError("CANCELLED_BY_USER")
        set_progress(task, "running", f"Step {step + 1}/{total_steps}", step + 1, total_steps)
        return kwargs
    return callback


@router.post("/cancel")
async def cancel_inference():
    """Request cancellation of the current inference."""
    request_cancel()
    return {"status": "cancel_requested"}


@router.get("/progress")
async def get_progress():
    """Return current inference progress."""
    if PROGRESS["started_at"] > 0 and PROGRESS["active"]:
        PROGRESS["elapsed"] = round(time.time() - PROGRESS["started_at"], 1)
    return PROGRESS


@router.get("/vram")
async def vram_status():
    return get_vram_info()


@router.get("/health")
async def health():
    return {"status": "ok"}


@router.post("/clear/soft")
async def clear_soft():
    return soft_clear_vram()


@router.post("/clear/hard")
async def clear_hard():
    result = hard_clear_vram()
    return {**result, "vram": get_vram_info()}


@router.post("/clear/aux")
async def clear_aux():
    unload_aux_pipelines()
    return {"message": "Auxiliary pipelines unloaded.", "vram": get_vram_info()}


@router.post("/enrich")
async def enrich_preview(data: dict):
    """Preview enriched prompt (for Edit page)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from prompt_enricher import enrich_positive, enrich_negative, get_preset_list
    prompt = data.get("prompt", "")
    negative = data.get("negative", "")
    auto = data.get("auto_enrich", True)
    preset = data.get("preset", "general")
    if auto:
        pos, info = enrich_positive(prompt, preset=preset)
        neg = enrich_negative(negative, preset=preset)
    else:
        pos = prompt
        neg = negative or ""
    return {"positive": pos, "negative": neg}


@router.get("/presets")
async def list_presets():
    """Return available enricher presets for frontend."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from prompt_enricher import get_preset_list
    return {"presets": get_preset_list()}


@router.get("/presets/gen")
async def list_gen_presets():
    """Return available generation presets (POV, Composition)."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from prompt_enricher import get_gen_preset_list
    return {"presets": get_gen_preset_list()}


FLUX_PHOTO_TOKENS = [
    "professional photography", "shot on Canon EOS R5", "85mm f/1.4",
    "natural studio lighting", "soft rim light", "shallow depth of field",
    "RAW photo", "ultra high resolution", "8k uhd", "film grain",
    "photorealistic", "hyperrealistic", "lifelike skin texture",
    "natural skin pores", "subsurface scattering", "fine body hair",
    "natural eye reflections", "catchlight in eyes",
    "realistic hair strands", "volumetric hair",
    "natural facial proportions", "anatomically correct",
    "sharp focus on face", "bokeh background",
]

FLUX_PHOTO_NEGATIVE = (
    "cartoon, anime, drawing, painting, illustration, 3d render, CGI, "
    "plastic skin, over-smoothed, airbrushed, uncanny valley, "
    "bad anatomy, deformed, extra fingers, missing fingers, "
    "text, watermark, logo, blurry, low quality"
)


@router.post("/enrich/flux")
async def api_enrich_flux(data: dict):
    """Enrich prompt for FLUX using selected generation preset."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from prompt_enricher import enrich_flux
    prompt = data.get("prompt", "")
    auto = data.get("auto_enrich", True)
    preset = data.get("preset", "general")
    if auto:
        enriched, negative = enrich_flux(prompt, preset=preset)
    else:
        enriched = prompt
        negative = ""
    return {"positive": enriched, "negative": negative}

