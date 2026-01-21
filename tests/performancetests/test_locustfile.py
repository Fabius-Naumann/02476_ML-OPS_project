"""Locust load tests for the Traffic Sign Inference API.

This script defines a user behavior that exercises:
- GET /, GET /health, GET /model
- POST /predict with a small in-memory PNG image
- Admin reads: GET /admin/status, GET /admin/train, GET /admin/evaluate
- Optional admin actions: POST /admin/train_sync, /admin/evaluate_sync, /admin/test_sync
"""

from __future__ import annotations

import io
import os
import sys
from typing import Final

from PIL import Image


def _performance_tests_enabled() -> bool:
    return os.getenv("SIGN_ML_RUN_PERFORMANCE_TESTS", "0").strip() not in {"0", "false", "False"}


_IS_PYTEST = "pytest" in sys.modules


def _pytest_skip(reason: str) -> None:
    import pytest

    pytest.skip(reason, allow_module_level=True)


if _IS_PYTEST and sys.platform == "win32":
    _pytest_skip("Locust performance tests are not supported on Windows.")

if _IS_PYTEST and not _performance_tests_enabled():
    _pytest_skip("Performance tests disabled (set SIGN_ML_RUN_PERFORMANCE_TESTS=1 to enable).")


from locust import HttpUser, between, task  # noqa: E402


def _png_bytes(*, size: tuple[int, int] = (8, 8), color: tuple[int, int, int] = (255, 0, 0)) -> bytes:
    image = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


_PNG_BYTES: Final[bytes] = _png_bytes()


class ApiUser(HttpUser):
    """Simulated API user performing basic read and inference requests."""

    wait_time = between(0.1, 0.5)

    _ADMIN_SYNC_ENABLED: Final[bool] = os.getenv("LOCUST_ENABLE_ADMIN_SYNC", "0").strip() not in {
        "0",
        "false",
        "False",
    }

    @task(1)
    def read_root(self) -> None:
        self.client.get("/")

    @task(2)
    def read_health(self) -> None:
        self.client.get("/health")

    @task(1)
    def read_model_info(self) -> None:
        self.client.get("/model")

    @task(3)
    def predict_small_image(self) -> None:
        files = {"image": ("x.png", _PNG_BYTES, "image/png")}
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"{response.status_code}: {response.text}")

    @task(1)
    def read_admin_status(self) -> None:
        self.client.get("/admin/status")

    @task(1)
    def read_admin_train_latest(self) -> None:
        self.client.get("/admin/train")

    @task(1)
    def read_admin_evaluate_latest(self) -> None:
        self.client.get("/admin/evaluate")

    @task(1)
    def admin_train_sync_small(self) -> None:
        if not self._ADMIN_SYNC_ENABLED:
            return
        self.client.post(
            "/admin/train_sync",
            params={"epochs": 5, "batch_size": 64, "lr": 0.001},
        )

    @task(1)
    def admin_evaluate_sync_small(self) -> None:
        if not self._ADMIN_SYNC_ENABLED:
            return
        self.client.post(
            "/admin/evaluate_sync",
            params={"batch_size": 64},
        )

    @task(1)
    def admin_test_sync(self) -> None:
        if not self._ADMIN_SYNC_ENABLED:
            return
        self.client.post("/admin/test_sync")
