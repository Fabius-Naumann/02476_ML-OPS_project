"""Locust load tests for the Traffic Sign Inference API.

This script defines a user behavior that exercises:
- GET /, GET /health, GET /model
- POST /predict with a small in-memory PNG image
- Admin reads: GET /admin/status, GET /admin/train, GET /admin/evaluate
- Optional admin actions: POST /admin/train_sync, /admin/evaluate_sync, /admin/test_sync
"""

from __future__ import annotations

import base64
import os
from typing import Final

from locust import HttpUser, between, task

_ONE_BY_ONE_PNG_B64: Final[str] = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9ySFTN8AAAAASUVORK5CYII="
)
_PNG_BYTES: Final[bytes] = base64.b64decode(_ONE_BY_ONE_PNG_B64)


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
        self.client.post("/predict", files=files)

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
