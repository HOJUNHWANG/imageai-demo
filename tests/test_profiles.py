from backend.core.config import EDIT_PROFILES, GENERATE_PROFILES, get_profile
from backend.routers.studio import GenerateRequest, _fit_dimensions


def test_profiles_have_distinct_runtime_strategies() -> None:
    assert [GENERATE_PROFILES[key].family for key in ("quality", "balanced", "fast")] == [
        "flux",
        "zimage",
        "zimage",
    ]
    assert [EDIT_PROFILES[key].steps for key in ("quality", "balanced", "fast")] == [40, 8, 4]
    assert EDIT_PROFILES["balanced"].lora_id
    assert EDIT_PROFILES["fast"].transformer_id
    assert get_profile("generate", "balanced").prequantized is True


def test_generate_request_defaults_to_fast() -> None:
    request = GenerateRequest(prompt="portrait")
    assert request.profile == "fast"


def test_fast_pixel_budget_preserves_aspect_ratio_approximately() -> None:
    width, height = _fit_dimensions(1536, 1024, 896 * 896)
    assert width * height <= 896 * 896
    assert abs((width / height) - 1.5) < 0.08
