from fastapi.testclient import TestClient

from backend.main import app


def test_runtime_endpoints_do_not_load_models() -> None:
    with TestClient(app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        config = client.get("/api/config").json()
        status = client.get("/api/status").json()

    assert config["safety_checker"] is False
    assert config["default_profile"] == "fast"
    assert set(config["profiles"]["generate"]) == {"quality", "balanced", "fast"}
    assert set(config["profiles"]["edit"]) == {"quality", "balanced", "fast"}
    assert config["profiles"]["generate"]["quality"]["model_id"] == "kpsss34/FHDR_Uncensored"
    assert config["profiles"]["edit"]["fast"]["steps"] == 4
    assert config["profiles"]["generate"]["fast"]["download_gb"] > 0
    assert isinstance(config["profiles"]["generate"]["fast"]["cached"], bool)
    assert config["free_disk_gb"] > 0
    assert config["max_upload_mb"] == 32
    assert isinstance(config["hf_token_configured"], bool)
    assert status["model"]["loaded"] is False
    assert status["hardware"]["device"] in {"cpu", "cuda"}
    if status["hardware"]["device"] == "cuda":
        assert status["hardware"]["gpu_name"]
    assert status["job"]["overall_progress"] >= 0
    assert "eta_seconds" in status["job"]
    assert "cancellable" in status["job"]


def test_cancel_without_active_job_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post("/api/cancel")

    assert response.status_code == 409
