"""Integration tests for the FastAPI application.

These tests exercise the API as a whole using FastAPI's TestClient.

Notes:
    The app uses a lifespan hook to load a model at startup. For stable tests
    (and to avoid depending on external model artifacts), most tests create a
    fresh app instance via `create_app()` and monkeypatch the model-loading
    function to return a small dummy model.
"""

from __future__ import annotations

import io
import types

import pytest
import torch
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

import sign_ml.api as api_module

"""Create a TestClient for a fresh app with a monkeypatched dummy model.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        num_classes: Number of classes to have the dummy model output.

    Returns:
        A TestClient bound to a fresh FastAPI app instance.
"""


def _make_test_client_with_dummy_model(monkeypatch: pytest.MonkeyPatch, *, num_classes: int = 3) -> TestClient:
    class DummyModel(torch.nn.Module):
        def __init__(self, classes: int) -> None:
            super().__init__()
            self._classes = classes

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            batch = int(x.shape[0])
            return torch.zeros((batch, self._classes), dtype=torch.float32)

    def fake_load_model(model_path: object, device: torch.device) -> tuple[torch.nn.Module, int]:
        _ = model_path
        _ = device
        return DummyModel(num_classes), num_classes

    monkeypatch.setattr(api_module, "_load_model", fake_load_model)
    test_app = api_module.create_app()
    return TestClient(test_app)


"""Create a tiny valid PNG image as bytes."""


def _png_bytes(*, size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


"""GET / returns a simple usage hint."""


def test_root_returns_usage_message(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Use GET /health for status and POST /predict for inference."}


"""GET /health returns a JSON payload with expected keys."""


def test_health_has_expected_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "is_loaded" in payload
    assert "weights_file" in payload
    assert "num_classes" in payload


"""POST /predict rejects empty uploads."""


def test_predict_empty_file_returns_400(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.post("/predict", files={"image": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert "Empty file" in response.json().get("detail", "")


"""POST /predict enforces a maximum upload size."""


def test_predict_large_upload_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGN_ML_MAX_UPLOAD_BYTES", "1024")
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.post("/predict", files={"image": ("big.png", b"x" * 2048, "image/png")})
    assert response.status_code == 413


"""POST /predict rejects non-image content types with 415."""


def test_predict_rejects_non_image_content_type(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.post("/predict", files={"image": ("file.txt", b"hello", "text/plain")})
    assert response.status_code == 415


"""POST /predict rejects corrupt image bytes with 400."""


def test_predict_rejects_invalid_image_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.post("/predict", files={"image": ("bad.png", b"not-a-png", "image/png")})
    assert response.status_code == 400


"""POST /predict returns probabilities and a predicted class on success."""


def test_predict_success_returns_expected_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch, num_classes=5)
    with client:
        response = client.post("/predict", files={"image": ("x.png", _png_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["num_classes"] == 5
    assert isinstance(payload["predicted_class"], int)
    assert isinstance(payload["probabilities"], list)
    assert len(payload["probabilities"]) == 5


"""GET /model returns metadata."""


def test_model_info_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.get("/model")
    assert response.status_code == 200
    payload = response.json()
    assert "model_path" in payload
    assert "num_classes" in payload
    assert "device" in payload


"""GET /admin/status reports current job counts."""


def test_admin_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        response = client.get("/admin/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert isinstance(payload["max_running_jobs"], int)
    assert isinstance(payload["running_jobs"], int)
    assert isinstance(payload["total_jobs"], int)


"""GET /admin/train and /admin/evaluate return not_started when no jobs exist."""


def test_admin_latest_endpoints_default_to_not_started(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)
    with client:
        train = client.get("/admin/train")
        evaluate = client.get("/admin/evaluate")
    assert train.status_code == 200
    assert evaluate.status_code == 200
    assert train.json()["status"] in {"not_started", "from_outputs"}
    assert evaluate.json()["status"] in {"not_started", "from_outputs"}


"""POST /admin/train_sync returns a job payload without spawning subprocesses."""


def test_admin_train_sync_is_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)

    def start_admin_job(*, action: str, args: list[str]) -> str:
        _ = action
        _ = args
        return "job-123"

    def wait_for_admin_job(job: object, *, timeout_seconds: float | None = None) -> None:
        _ = job
        _ = timeout_seconds

    monkeypatch.setattr(api_module, "_start_admin_job", start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", lambda job_id: types.SimpleNamespace(job_id=job_id))
    monkeypatch.setattr(api_module, "_wait_for_admin_job", wait_for_admin_job)
    monkeypatch.setattr(
        api_module,
        "_job_to_result",
        lambda job: api_module.JobResultResponse(
            job_id=job.job_id,
            action="train",
            status="completed",
            started_at="now",
            ended_at="now",
            return_code=0,
            log_tail=[],
        ),
    )

    with client:
        response = client.post("/admin/train_sync", params={"epochs": 1})
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-123"


"""POST /admin/train_sync surfaces 429 when job start is rejected."""


def test_admin_train_sync_can_return_429(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)

    def reject(*, action: str, args: list[str]) -> str:
        _ = action
        _ = args
        raise HTTPException(status_code=429, detail="Too many running jobs")

    monkeypatch.setattr(api_module, "_start_admin_job", reject)
    with client:
        response = client.post("/admin/train_sync")
    assert response.status_code == 429


"""POST /admin/evaluate_sync returns a job payload without spawning subprocesses."""


def test_admin_evaluate_sync_is_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)

    def start_admin_job(*, action: str, args: list[str]) -> str:
        _ = action
        _ = args
        return "job-456"

    def wait_for_admin_job(job: object, *, timeout_seconds: float | None = None) -> None:
        _ = job
        _ = timeout_seconds

    monkeypatch.setattr(api_module, "_start_admin_job", start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", lambda job_id: types.SimpleNamespace(job_id=job_id))
    monkeypatch.setattr(api_module, "_wait_for_admin_job", wait_for_admin_job)
    monkeypatch.setattr(
        api_module,
        "_job_to_result",
        lambda job: api_module.JobResultResponse(
            job_id=job.job_id,
            action="evaluate",
            status="completed",
            started_at="now",
            ended_at="now",
            return_code=0,
            log_tail=[],
        ),
    )

    with client:
        response = client.post("/admin/evaluate_sync", params={"batch_size": 1})
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-456"


"""POST /admin/test_sync returns a job payload without running pytest subprocesses."""


def test_admin_test_sync_is_stubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _make_test_client_with_dummy_model(monkeypatch)

    def start_admin_job(*, action: str, args: list[str]) -> str:
        _ = action
        _ = args
        return "job-789"

    def wait_for_admin_job(job: object, *, timeout_seconds: float | None = None) -> None:
        _ = job
        _ = timeout_seconds

    monkeypatch.setattr(api_module, "_start_admin_job", start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", lambda job_id: types.SimpleNamespace(job_id=job_id))
    monkeypatch.setattr(api_module, "_wait_for_admin_job", wait_for_admin_job)
    monkeypatch.setattr(
        api_module,
        "_job_to_result",
        lambda job: api_module.JobResultResponse(
            job_id=job.job_id,
            action="test",
            status="completed",
            started_at="now",
            ended_at="now",
            return_code=0,
            log_tail=[],
        ),
    )

    with client:
        response = client.post("/admin/test_sync")
    assert response.status_code == 200
    assert response.json()["job_id"] == "job-789"


"""POST /predict returns 503 if the model failed to load during lifespan."""


def test_predict_returns_503_when_model_not_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIGN_ML_MODEL_PATH", "models/does_not_exist.pt")
    test_app = api_module.create_app()
    with TestClient(test_app) as client:
        health = client.get("/health")
        response = client.post("/predict", files={"image": ("x.png", _png_bytes(), "image/png")})
    assert health.status_code == 200
    assert health.json()["is_loaded"] is False
    assert response.status_code == 503
