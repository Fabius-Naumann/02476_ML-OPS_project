"""FastAPI application for traffic sign inference.

This module exposes a small REST API that can be used to run inference on the
trained PyTorch model.
"""

from __future__ import annotations

import datetime
import io
import os
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from sign_ml import BASE_DIR
from sign_ml.model import build_model

DEFAULT_MODEL_PATH = BASE_DIR / "models" / "traffic_sign_model.pt"
IMAGE_FILE = File(...)


def _max_upload_bytes() -> int:
    raw = os.getenv("SIGN_ML_MAX_UPLOAD_BYTES", "5242880")
    try:
        value = int(raw)
    except ValueError:
        return 1048576
    return max(1024, value)


def _max_image_pixels() -> int:
    raw = os.getenv("SIGN_ML_MAX_IMAGE_PIXELS", "10000000")
    try:
        value = int(raw)
    except ValueError:
        return 10000000
    return max(65536, value)


async def _read_upload_file_limited(upload: UploadFile, *, max_bytes: int) -> bytes:
    total = 0
    chunks: list[bytes] = []
    # Read in 1 MiB chunks up to max_bytes
    while True:
        to_read = min(1024 * 1024, max_bytes - total)
        if to_read <= 0:
            raise HTTPException(status_code=413, detail=f"File too large (> {max_bytes} bytes)")
        chunk = await upload.read(to_read)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large (> {max_bytes} bytes)")
    return b"".join(chunks)


def _max_admin_jobs() -> int:
    raw = os.getenv("SIGN_ML_MAX_ADMIN_JOBS", "10")
    try:
        value = int(raw)
    except ValueError:
        return 1
    return max(1, value)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _admin_log_dir() -> Path:
    path = BASE_DIR / "log" / "api_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _outputs_dir() -> Path:
    return BASE_DIR / "outputs"


def _include_outputs_fallback() -> bool:
    return os.getenv("SIGN_ML_ADMIN_FALLBACK_TO_OUTPUTS", "1").strip() not in {"0", "false", "False"}


def _latest_outputs_log_path(action: str) -> Path | None:
    base = _outputs_dir()
    if not base.exists():
        return None

    candidates = list(base.glob(f"**/{action}.log"))
    if not candidates:
        return None

    def key(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    newest = max(candidates, key=key)
    return newest if newest.exists() else None


def _read_log_tail(path: Path, *, max_lines: int = 200) -> list[str]:
    if max_lines <= 0:
        return []
    if not path.exists():
        return []
    try:
        from collections import deque

        with path.open("r", encoding="utf-8", errors="replace") as f:
            tail: deque[str] = deque(maxlen=max_lines)
            for line in f:
                tail.append(line.rstrip("\n"))
        return list(tail)
    except OSError:
        return []


@dataclass
class _AdminJob:
    job_id: str
    action: str
    command: list[str]
    cwd: Path
    log_path: Path
    process: subprocess.Popen[str]
    started_at: datetime.datetime
    ended_at: datetime.datetime | None = None
    return_code: int | None = None

    def refresh(self) -> None:
        if self.return_code is not None:
            return
        code = self.process.poll()
        if code is None:
            return
        self.return_code = int(code)
        self.ended_at = datetime.datetime.now(tz=datetime.UTC)

    def status(self) -> str:
        self.refresh()
        if self.return_code is None:
            return "running"
        return "completed" if self.return_code == 0 else "failed"


_ADMIN_JOBS: dict[str, _AdminJob] = {}
_ADMIN_JOBS_LOCK = threading.Lock()
_ADMIN_JOBS_RESERVED: int = 0


def _running_admin_jobs_count() -> int:
    with _ADMIN_JOBS_LOCK:
        running = 0
        for job in _ADMIN_JOBS.values():
            if job.process.poll() is None:
                running += 1
        return running


def _start_admin_job(*, action: str, args: list[str]) -> str:
    """Start a background admin job, enforcing max concurrency under lock.

    Uses a reservation counter to prevent races between the concurrency check
    and job registration, ensuring that concurrent requests cannot exceed the
    configured limit.
    """

    global _ADMIN_JOBS_RESERVED

    max_jobs = _max_admin_jobs()

    # Reserve a slot under lock to avoid races with concurrent starters.
    with _ADMIN_JOBS_LOCK:
        running = 0
        for job in _ADMIN_JOBS.values():
            if job.process.poll() is None:
                running += 1

        if running + _ADMIN_JOBS_RESERVED >= max_jobs:
            raise HTTPException(status_code=429, detail=f"Too many running jobs (limit={max_jobs})")

        _ADMIN_JOBS_RESERVED += 1

    job_id = uuid.uuid4().hex
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d-%H%M%S")
    log_path = _admin_log_dir() / f"{timestamp}-{action}-{job_id}.log"
    cwd = _repo_root()

    cmd = [sys.executable, *args]

    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PROJECT_ROOT", str(BASE_DIR))

    log_file = log_path.open("w", encoding="utf-8")
    try:
        process: subprocess.Popen[str] = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        # If starting the process fails, release the reserved slot.
        with _ADMIN_JOBS_LOCK:
            _ADMIN_JOBS_RESERVED = max(0, _ADMIN_JOBS_RESERVED - 1)
        raise
    finally:
        log_file.close()

    job = _AdminJob(
        job_id=job_id,
        action=action,
        command=cmd,
        cwd=cwd,
        log_path=log_path,
        process=process,
        started_at=datetime.datetime.now(tz=datetime.UTC),
    )

    # Register the job and release the reservation.
    with _ADMIN_JOBS_LOCK:
        _ADMIN_JOBS[job_id] = job
        _ADMIN_JOBS_RESERVED = max(0, _ADMIN_JOBS_RESERVED - 1)

    logger.info("Started admin job {}: {}", job_id, " ".join(cmd))
    return job_id


def _get_admin_job(job_id: str) -> _AdminJob:
    with _ADMIN_JOBS_LOCK:
        job = _ADMIN_JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


def _latest_admin_job(action: str) -> _AdminJob | None:
    with _ADMIN_JOBS_LOCK:
        candidates = [job for job in _ADMIN_JOBS.values() if job.action == action]
    if not candidates:
        return None
    return max(candidates, key=lambda job: job.started_at)


def _wait_for_admin_job(job: _AdminJob, *, timeout_seconds: int | None) -> None:
    if job.process.poll() is not None:
        job.refresh()
        return
    if timeout_seconds is not None and timeout_seconds <= 0:
        job.refresh()
        return
    try:
        job.process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        job.refresh()


def _get_admin_sync_timeout_seconds() -> int:
    raw = os.getenv("SIGN_ML_ADMIN_SYNC_TIMEOUT_SECONDS")
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return 600
        return max(1, value)

    raw = os.getenv("SIGN_ML_ADMIN_TRAIN_TIMEOUT_SECONDS", "600")
    try:
        value = int(raw)
    except ValueError:
        return 600
    return max(1, value)


def _get_admin_log_tail_lines() -> int:
    raw = os.getenv("SIGN_ML_ADMIN_LOG_TAIL_LINES", "200")
    try:
        value = int(raw)
    except ValueError:
        return 200
    return max(1, value)


class HealthResponse(BaseModel):
    """Health status response."""

    status: str
    is_loaded: bool
    weights_file: str
    num_classes: int | None
    detail: str | None = None


class PredictResponse(BaseModel):
    """Prediction response."""

    predicted_class: int
    probabilities: list[float]
    num_classes: int


@dataclass
class _ModelState:
    model: torch.nn.Module | None = None
    num_classes: int | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_path: Path = DEFAULT_MODEL_PATH
    load_error: str | None = None


class AdminStatusResponse(BaseModel):
    """Admin endpoints status."""

    enabled: bool
    max_running_jobs: int
    running_jobs: int
    total_jobs: int


class JobStartResponse(BaseModel):
    """Response returned when a background job is started."""

    job_id: str


class JobResultResponse(BaseModel):
    """Result of a background job (or current status if still running)."""

    job_id: str
    action: str
    status: str
    started_at: str
    ended_at: str | None
    return_code: int | None
    log_tail: list[str]


class LatestJobResponse(BaseModel):
    """Latest known job information for an action (does not start a new job)."""

    action: str
    status: str
    job_id: str | None
    started_at: str | None
    ended_at: str | None
    return_code: int | None
    log_path: str | None
    log_tail: list[str]


def _job_to_result(job: _AdminJob) -> JobResultResponse:
    return JobResultResponse(
        job_id=job.job_id,
        action=job.action,
        status=job.status(),
        started_at=job.started_at.isoformat(),
        ended_at=job.ended_at.isoformat() if job.ended_at else None,
        return_code=job.return_code,
        log_tail=_read_log_tail(job.log_path, max_lines=_get_admin_log_tail_lines()),
    )


def _latest_job_response(action: str) -> LatestJobResponse:
    job = _latest_admin_job(action)
    if job is None:
        if _include_outputs_fallback():
            outputs_log = _latest_outputs_log_path(action)
            if outputs_log is not None:
                try:
                    ts = datetime.datetime.fromtimestamp(outputs_log.stat().st_mtime, tz=datetime.UTC).isoformat()
                except OSError:
                    ts = None
                return LatestJobResponse(
                    action=action,
                    status="from_outputs",
                    job_id=None,
                    started_at=ts,
                    ended_at=ts,
                    return_code=None,
                    log_path=str(outputs_log),
                    log_tail=_read_log_tail(outputs_log, max_lines=_get_admin_log_tail_lines()),
                )

        return LatestJobResponse(
            action=action,
            status="not_started",
            job_id=None,
            started_at=None,
            ended_at=None,
            return_code=None,
            log_path=None,
            log_tail=[],
        )

    job.refresh()
    return LatestJobResponse(
        action=job.action,
        status=job.status(),
        job_id=job.job_id,
        started_at=job.started_at.isoformat(),
        ended_at=job.ended_at.isoformat() if job.ended_at else None,
        return_code=job.return_code,
        log_path=str(job.log_path),
        log_tail=_read_log_tail(job.log_path, max_lines=_get_admin_log_tail_lines()),
    )


def _resolve_model_path() -> Path:
    raw = os.getenv("SIGN_ML_MODEL_PATH")
    if not raw:
        return DEFAULT_MODEL_PATH
    path = Path(raw)
    return path if path.is_absolute() else BASE_DIR / path


def _build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def _infer_num_classes_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    fc_weight = state_dict.get("backbone.fc.weight")
    if fc_weight is None or fc_weight.ndim != 2:
        raise ValueError("Unable to infer num_classes: expected key 'backbone.fc.weight' in state_dict")
    return int(fc_weight.shape[0])


def _load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if not isinstance(state_dict, dict):
        raise TypeError("Expected a state_dict (dict) in model file")

    num_classes = _infer_num_classes_from_state_dict(state_dict)

    model = build_model(num_classes)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning("Model state_dict mismatch. Missing keys: {}. Unexpected keys: {}", missing, unexpected)

    model.to(device)
    model.eval()
    return model, num_classes


def create_app() -> FastAPI:  # noqa: C901
    """Create and configure the FastAPI application."""

    state = _ModelState(model_path=_resolve_model_path())
    preprocess = _build_preprocess()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state.device = torch.device(os.getenv("SIGN_ML_DEVICE", "cpu"))
            state.model_path = _resolve_model_path()
            state.model, state.num_classes = _load_model(state.model_path, state.device)
            state.load_error = None
            logger.info(
                "Loaded model from {} (classes={}, device={})", state.model_path, state.num_classes, state.device
            )
        except Exception as exc:
            state.model = None
            state.num_classes = None
            state.load_error = str(exc)
            logger.error("Failed to load model: {}", exc)

        yield

    app = FastAPI(title="Traffic Sign Inference API", version="0.1.0", lifespan=lifespan)

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        """Root endpoint with a short usage hint."""

        return {"message": "Use GET /health for status and POST /predict for inference."}

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """Health check endpoint."""

        model_loaded = state.model is not None
        return HealthResponse(
            status="ok" if model_loaded else "not_ready",
            is_loaded=model_loaded,
            weights_file=str(state.model_path),
            num_classes=state.num_classes,
            detail=state.load_error,
        )

    @app.get("/model", tags=["meta"])
    def model_info() -> dict[str, str | int | None]:
        """Return basic model metadata."""

        return {
            "model_path": str(state.model_path),
            "num_classes": state.num_classes,
            "device": str(state.device),
        }

    @app.get("/admin/status", response_model=AdminStatusResponse, tags=["admin"])
    def admin_status() -> AdminStatusResponse:
        """Return whether admin endpoints are enabled and current job counts."""

        with _ADMIN_JOBS_LOCK:
            total = len(_ADMIN_JOBS)
        return AdminStatusResponse(
            enabled=True,
            max_running_jobs=_max_admin_jobs(),
            running_jobs=_running_admin_jobs_count(),
            total_jobs=total,
        )

    @app.get("/admin/train", response_model=LatestJobResponse, tags=["admin"])
    def admin_train_latest() -> LatestJobResponse:
        """Return information about the latest train job (does not start a new job)."""

        return _latest_job_response("train")

    @app.post("/admin/train_sync", response_model=JobResultResponse, tags=["admin"])
    def admin_train_sync(
        epochs: int | None = None,
        batch_size: int | None = None,
        lr: float | None = None,
    ) -> JobResultResponse:
        """Run training and return the job result (or running status on timeout)."""

        overrides = ["hydra.job.chdir=false"]
        if epochs is not None:
            overrides.append(f"experiment.training.epochs={epochs}")
        if batch_size is not None:
            overrides.append(f"experiment.training.batch_size={batch_size}")
        if lr is not None:
            overrides.append(f"experiment.optimizer.lr={lr}")

        job_id = _start_admin_job(action="train", args=["-m", "sign_ml.train", *overrides])
        job = _get_admin_job(job_id)
        _wait_for_admin_job(job, timeout_seconds=_get_admin_sync_timeout_seconds())
        return _job_to_result(job)

    @app.get("/admin/evaluate", response_model=LatestJobResponse, tags=["admin"])
    def admin_evaluate_latest() -> LatestJobResponse:
        """Return information about the latest evaluate job (does not start a new job)."""

        return _latest_job_response("evaluate")

    @app.post("/admin/evaluate_sync", response_model=JobResultResponse, tags=["admin"])
    def admin_evaluate_sync(batch_size: int | None = None) -> JobResultResponse:
        """Run evaluation and return the job result (or running status on timeout)."""

        overrides = ["hydra.job.chdir=false"]
        if batch_size is not None:
            overrides.append(f"experiment.training.batch_size={batch_size}")

        job_id = _start_admin_job(action="evaluate", args=["-m", "sign_ml.evaluate", *overrides])
        job = _get_admin_job(job_id)
        _wait_for_admin_job(job, timeout_seconds=_get_admin_sync_timeout_seconds())
        return _job_to_result(job)

    @app.post("/admin/test_sync", response_model=JobResultResponse, tags=["admin"])
    def admin_test_sync() -> JobResultResponse:
        """Run the repo's test suite and return the result (or running status on timeout)."""

        job_id = _start_admin_job(action="test", args=["-m", "pytest", "-q", "tests/"])
        job = _get_admin_job(job_id)
        _wait_for_admin_job(job, timeout_seconds=_get_admin_sync_timeout_seconds())
        return _job_to_result(job)

    @app.post("/predict", response_model=PredictResponse, tags=["inference"])
    async def predict(image: UploadFile = IMAGE_FILE) -> PredictResponse:
        """Run model inference on a single uploaded image."""

        if state.model is None or state.num_classes is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if image.content_type is None or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Unsupported media type; expected an image")

        # Enforce explicit max upload size before decoding
        content = await _read_upload_file_limited(image, max_bytes=_max_upload_bytes())
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        try:
            # Guard against decompression bombs by limiting pixel count
            Image.MAX_IMAGE_PIXELS = _max_image_pixels()
            pil_image = Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

        x = preprocess(pil_image).unsqueeze(0)
        x = x.to(state.device)

        with torch.inference_mode():
            logits = state.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            predicted_class = int(torch.argmax(probs).item())

        return PredictResponse(
            predicted_class=predicted_class,
            probabilities=[float(p) for p in probs.detach().cpu().tolist()],
            num_classes=state.num_classes,
        )

    return app


app = create_app()