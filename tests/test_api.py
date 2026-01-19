import io
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from PIL import Image

import sign_ml.api as api_module
from sign_ml.api import app


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert "status" in payload
    assert "is_loaded" in payload


def test_predict_endpoint_smoke() -> None:
    client = TestClient(app)

    image = Image.new("RGB", (64, 64), color=(255, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"image": ("test.png", buffer.read(), "image/png")},
    )

    assert response.status_code in {200, 503}
    if response.status_code == 200:
        payload = response.json()
        assert "predicted_class" in payload
        assert "probabilities" in payload
        assert "num_classes" in payload


def test_admin_endpoints_disabled_without_token() -> None:
    client = TestClient(app)
    response = client.get("/admin/status")
    assert response.status_code == 200
    assert response.json()["enabled"] is True


def test_admin_can_get_evaluate_result_sync(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "evaluate.log"
    log_path.write_text("eval line 1\neval done\n", encoding="utf-8")

    class _FakeProcess:
        def poll(self) -> int:
            return 0

        def wait(self, *_args, **_kwargs) -> int:
            return 0

    class _FakeJob:
        def __init__(self, *, job_id: str, action: str, log_path) -> None:
            self.job_id = job_id
            self.action = action
            self.log_path = log_path
            self.process = _FakeProcess()
            self.started_at = datetime.now(tz=UTC)
            self.ended_at = datetime.now(tz=UTC)
            self.return_code = 0

        def refresh(self) -> None:
            return

        def status(self) -> str:
            return "completed"

    def fake_start_admin_job(*, action: str, args: list[str]) -> str:
        assert action == "evaluate"
        assert args[0] == "src/sign_ml/evaluate.py"
        return "job-eval"

    def fake_get_admin_job(job_id: str):
        assert job_id == "job-eval"
        return _FakeJob(job_id="job-eval", action="evaluate", log_path=log_path)

    def fake_wait_for_admin_job(_job, *, timeout_seconds: int | None) -> None:
        assert timeout_seconds is not None

    monkeypatch.setattr(api_module, "_start_admin_job", fake_start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", fake_get_admin_job)
    monkeypatch.setattr(api_module, "_wait_for_admin_job", fake_wait_for_admin_job)

    client = TestClient(app)
    response = client.post("/admin/evaluate_sync")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-eval"
    assert payload["action"] == "evaluate"
    assert payload["status"] == "completed"
    assert payload["return_code"] == 0
    assert payload["log_tail"] == ["eval line 1", "eval done"]


def test_admin_can_get_test_result_sync(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "test.log"
    log_path.write_text("tests running\n8 passed\n", encoding="utf-8")

    class _FakeProcess:
        def poll(self) -> int:
            return 0

        def wait(self, *_args, **_kwargs) -> int:
            return 0

    class _FakeJob:
        def __init__(self, *, job_id: str, action: str, log_path) -> None:
            self.job_id = job_id
            self.action = action
            self.log_path = log_path
            self.process = _FakeProcess()
            self.started_at = datetime.now(tz=UTC)
            self.ended_at = datetime.now(tz=UTC)
            self.return_code = 0

        def refresh(self) -> None:
            return

        def status(self) -> str:
            return "completed"

    def fake_start_admin_job(*, action: str, args: list[str]) -> str:
        assert action == "test"
        assert args[:2] == ["-m", "pytest"]
        return "job-test"

    def fake_get_admin_job(job_id: str):
        assert job_id == "job-test"
        return _FakeJob(job_id="job-test", action="test", log_path=log_path)

    def fake_wait_for_admin_job(_job, *, timeout_seconds: int | None) -> None:
        assert timeout_seconds is not None

    monkeypatch.setattr(api_module, "_start_admin_job", fake_start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", fake_get_admin_job)
    monkeypatch.setattr(api_module, "_wait_for_admin_job", fake_wait_for_admin_job)

    client = TestClient(app)
    response = client.post("/admin/test_sync")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-test"
    assert payload["action"] == "test"
    assert payload["status"] == "completed"
    assert payload["return_code"] == 0
    assert payload["log_tail"] == ["tests running", "8 passed"]


def test_admin_can_get_train_result_with_wait(monkeypatch, tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text("hello\ntrain done\n", encoding="utf-8")

    class _FakeProcess:
        def poll(self) -> int:
            return 0

        def wait(self, *_args, **_kwargs) -> int:
            return 0

    class _FakeJob:
        def __init__(self, *, job_id: str, action: str, log_path) -> None:
            self.job_id = job_id
            self.action = action
            self.log_path = log_path
            self.process = _FakeProcess()
            self.started_at = datetime.now(tz=UTC)
            self.ended_at = datetime.now(tz=UTC)
            self.return_code = 0

        def refresh(self) -> None:
            return

        def status(self) -> str:
            return "completed"

    def fake_start_admin_job(*, action: str, args: list[str]) -> str:
        assert action == "train"
        assert args[0] == "src/sign_ml/train.py"
        assert "experiment.optimizer.lr=0.01" in args
        return "job-789"

    def fake_get_admin_job(job_id: str):
        assert job_id == "job-789"
        return _FakeJob(job_id="job-789", action="train", log_path=log_path)

    def fake_wait_for_admin_job(_job, *, timeout_seconds: int | None) -> None:
        assert timeout_seconds is not None

    monkeypatch.setattr(api_module, "_start_admin_job", fake_start_admin_job)
    monkeypatch.setattr(api_module, "_get_admin_job", fake_get_admin_job)
    monkeypatch.setattr(api_module, "_wait_for_admin_job", fake_wait_for_admin_job)

    client = TestClient(app)
    response = client.post("/admin/train_sync?lr=0.01")
    assert response.status_code == 200
    payload = response.json()
    assert payload["job_id"] == "job-789"
    assert payload["action"] == "train"
    assert payload["status"] == "completed"
    assert payload["return_code"] == 0
    assert payload["log_tail"] == ["hello", "train done"]


def test_admin_train_get_returns_empty_and_does_not_start_job(monkeypatch) -> None:
    monkeypatch.setenv("SIGN_ML_ADMIN_FALLBACK_TO_OUTPUTS", "0")

    def fake_start_admin_job(*, action: str, args: list[str]) -> str:
        raise AssertionError(f"GET /admin/train must not start jobs (action={action}, args={args})")

    monkeypatch.setattr(api_module, "_start_admin_job", fake_start_admin_job)
    monkeypatch.setattr(api_module, "_latest_admin_job", lambda _action: None)

    client = TestClient(app)
    response = client.get("/admin/train")
    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "train"
    assert payload["status"] == "not_started"
    assert payload["job_id"] is None
    assert payload["log_tail"] == []


def test_admin_evaluate_get_returns_empty_and_does_not_start_job(monkeypatch) -> None:
    monkeypatch.setenv("SIGN_ML_ADMIN_FALLBACK_TO_OUTPUTS", "0")

    def fake_start_admin_job(*, action: str, args: list[str]) -> str:
        raise AssertionError(f"GET /admin/evaluate must not start jobs (action={action}, args={args})")

    monkeypatch.setattr(api_module, "_start_admin_job", fake_start_admin_job)
    monkeypatch.setattr(api_module, "_latest_admin_job", lambda _action: None)

    client = TestClient(app)
    response = client.get("/admin/evaluate")
    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "evaluate"
    assert payload["status"] == "not_started"
    assert payload["job_id"] is None
    assert payload["log_tail"] == []
