from backend.core.runtime import JobState


def test_download_progress_is_honestly_indeterminate() -> None:
    job = JobState()
    job.begin("edit", "Starting")
    job.update("downloading", "Downloading model")

    snapshot = job.snapshot()
    assert snapshot["active"] is True
    assert snapshot["stage"] == "downloading"
    assert snapshot["indeterminate"] is True
    assert snapshot["eta_seconds"] is None


def test_inference_progress_uses_real_denoising_steps() -> None:
    job = JobState()
    job.begin("generate", "Starting")
    job.update("inference", "Encoding prompt", 0, 8)
    callback = job.callback(8)
    callback(None, 3, None, {"latents": None})

    snapshot = job.snapshot()
    assert snapshot["step"] == 4
    assert snapshot["total"] == 8
    assert snapshot["stage_progress"] == 0.5
    assert snapshot["overall_progress"] == 55.0
    assert snapshot["indeterminate"] is False
    assert snapshot["eta_seconds"] is not None

    job.finish("Done")
    finished = job.snapshot()
    assert finished["active"] is False
    assert finished["overall_progress"] == 100.0
