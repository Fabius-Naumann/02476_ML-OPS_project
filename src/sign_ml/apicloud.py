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
import gcsfs
from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from loguru import logger
from PIL import Image
from prometheus_client import generate_latest
from pydantic import BaseModel
from torchvision import transforms

from sign_ml import BASE_DIR
from sign_ml.model import build_model
from sign_ml.observability import log_prediction, metrics_middleware


IMAGE_FILE = File(...)
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "traffic_sign_model.pt"


# ------------------------------------------------------------------
# GCS / LOCAL MODEL RESOLUTION
# ------------------------------------------------------------------
def _resolve_model_path() -> Path:
    gcs_path = os.getenv("SIGN_ML_MODEL_GCS")
    raw_local = os.getenv("SIGN_ML_MODEL_PATH")

    local_path = DEFAULT_MODEL_PATH
    local_path.parent.mkdir(parents=True, exist_ok=True)

    if gcs_path:
        print(f"Loading model from GCS: {gcs_path}")
        fs = gcsfs.GCSFileSystem()
        with fs.open(gcs_path, "rb") as f:
            with open(local_path, "wb") as out:
                out.write(f.read())
        print(f"Model downloaded to {local_path}")
        return local_path

    if raw_local:
        path = Path(raw_local)
        resolved = path if path.is_absolute() else BASE_DIR / path
        print(f"Loading model from local path: {resolved}")
        return resolved

    print(f"Using default model path: {local_path}")
    return local_path


# ------------------------------------------------------------------
# PREPROCESS
# ------------------------------------------------------------------
def _build_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


# ------------------------------------------------------------------
# MODEL LOADER (.pt)
# ------------------------------------------------------------------
def _load_model(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    checkpoint = torch.load(model_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected checkpoint dict with state_dict and num_classes")

    state_dict = checkpoint["state_dict"]
    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, num_classes


# ------------------------------------------------------------------
# STATE
# ------------------------------------------------------------------
@dataclass
class _ModelState:
    model: torch.nn.Module | None = None
    num_classes: int | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_path: Path = DEFAULT_MODEL_PATH
    load_error: str | None = None


# ------------------------------------------------------------------
# APP FACTORY
# ------------------------------------------------------------------
def create_app() -> FastAPI:
    state = _ModelState()
    preprocess = _build_preprocess()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state.device = torch.device(os.getenv("SIGN_ML_DEVICE", "cpu"))
            state.model_path = _resolve_model_path()
            state.model, state.num_classes = _load_model(state.model_path, state.device)
            state.load_error = None
            logger.info(
                "Model loaded from {} (classes={}, device={})",
                state.model_path,
                state.num_classes,
                state.device,
            )
        except Exception as exc:
            state.model = None
            state.num_classes = None
            state.load_error = str(exc)
            logger.error("Failed to load model: {}", exc)

        yield

    app = FastAPI(title="Traffic Sign Inference API", version="1.0.0", lifespan=lifespan)
    app.middleware("http")(metrics_middleware)

    @app.get("/")
    def root():
        return {"message": "Use GET /health for status and POST /predict for inference."}

    @app.get("/health")
    def health():
        loaded = state.model is not None
        return {
            "status": "ok" if loaded else "not_ready",
            "is_loaded": loaded,
            "weights_file": str(state.model_path),
            "num_classes": state.num_classes,
            "detail": state.load_error,
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type="text/plain")

    @app.post("/predict")
    async def predict(image: UploadFile = IMAGE_FILE):
        if state.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=415, detail="Invalid image type")

        content = await image.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty image")

        img = Image.open(io.BytesIO(content)).convert("RGB")
        x = preprocess(img).unsqueeze(0).to(state.device)

        with torch.inference_mode():
            logits = state.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())

        log_prediction(
            input_summary={"filename": image.filename},
            output_summary={"predicted_class": pred},
        )

        return {
            "predicted_class": pred,
            "probabilities": [float(p) for p in probs.cpu().tolist()],
            "num_classes": state.num_classes,
        }

    return app


app = create_app()
