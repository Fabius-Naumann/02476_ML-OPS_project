"""Model gate: evaluate + lightweight performance benchmark.

This module is designed to be used in CI/CD when a new model artifact is produced.
It runs a quick evaluation on the test split and measures inference latency on CPU.

Supported model formats:
    - PyTorch state dict: .pt / .pth
    - Pickled PyTorch object/state dict: .pkl
    - ONNX: .onnx (via onnxruntime)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import typer
from loguru import logger

from sign_ml.data import TrafficSignsDataset
from sign_ml.model import build_model

if TYPE_CHECKING:
    import onnxruntime

app = typer.Typer(add_completion=False)

MODEL_PATH_OPTION = typer.Option(..., "--model-path", exists=True, dir_okay=False, readable=True)
REPORT_OPTION = typer.Option(Path("model_gate.md"), "--report")
BATCH_SIZE_OPTION = typer.Option(128, "--batch-size", min=1)
LATENCY_BATCH_SIZE_OPTION = typer.Option(1, "--latency-batch-size", min=1)
WARMUP_STEPS_OPTION = typer.Option(10, "--warmup-steps", min=0)
STEPS_OPTION = typer.Option(50, "--steps", min=1)


def _evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def _evaluate_onnx(
    *,
    session: onnxruntime.InferenceSession,
    loader: torch.utils.data.DataLoader,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()

    input_name = session.get_inputs()[0].name

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images_np = images.detach().cpu().numpy().astype("float32", copy=False)
            logits_np = session.run(None, {input_name: images_np})[0]

            logits = torch.from_numpy(logits_np)
            loss = criterion(logits, labels)

            total_loss += float(loss.item()) * images.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def _benchmark_latency_ms(
    model: nn.Module,
    *,
    batch_size: int,
    warmup_steps: int,
    steps: int,
    device: torch.device,
) -> tuple[float, float]:
    """Return (p50_ms, p95_ms) latency per batch."""

    model.eval()

    example = torch.randn(batch_size, 3, 64, 64, device=device)
    timings_s: list[float] = []

    with torch.inference_mode():
        for _ in range(warmup_steps):
            _ = model(example)

        for _ in range(steps):
            start = time.perf_counter()
            _ = model(example)
            end = time.perf_counter()
            timings_s.append(end - start)

    if not timings_s:
        return 0.0, 0.0

    timings_s.sort()
    p50 = timings_s[int(0.50 * (len(timings_s) - 1))]
    p95 = timings_s[int(0.95 * (len(timings_s) - 1))]
    return p50 * 1000.0, p95 * 1000.0


def _benchmark_latency_ms_onnx(
    session: onnxruntime.InferenceSession,
    *,
    batch_size: int,
    warmup_steps: int,
    steps: int,
) -> tuple[float, float]:
    """Return (p50_ms, p95_ms) latency per batch for ONNX runtime."""

    input_name = session.get_inputs()[0].name
    example = torch.randn(batch_size, 3, 64, 64).numpy().astype("float32", copy=False)
    timings_s: list[float] = []

    for _ in range(warmup_steps):
        _ = session.run(None, {input_name: example})

    for _ in range(steps):
        start = time.perf_counter()
        _ = session.run(None, {input_name: example})
        end = time.perf_counter()
        timings_s.append(end - start)

    if not timings_s:
        return 0.0, 0.0

    timings_s.sort()
    p50 = timings_s[int(0.50 * (len(timings_s) - 1))]
    p95 = timings_s[int(0.95 * (len(timings_s) - 1))]
    return p50 * 1000.0, p95 * 1000.0


def _write_report(
    report_path: Path,
    *,
    model_path: Path,
    model_format: str,
    test_loss: float,
    test_acc: float,
    p50_ms: float,
    p95_ms: float,
    batch_size: int,
    warmup_steps: int,
    steps: int,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = "\n".join(
        [
            "# Model gate report",
            "",
            f"- Model path: `{model_path}`",
            f"- Model format: `{model_format}`",
            f"- Test loss: `{test_loss:.4f}`",
            f"- Test accuracy: `{test_acc:.2f}%`",
            "",
            "## CPU latency benchmark",
            f"- Batch size: `{batch_size}`",
            f"- Warmup steps: `{warmup_steps}`",
            f"- Timed steps: `{steps}`",
            f"- p50 latency per batch: `{p50_ms:.2f} ms`",
            f"- p95 latency per batch: `{p95_ms:.2f} ms`",
            "",
        ]
    )

    report_path.write_text(report, encoding="utf-8")


def _load_torch_model_from_file(model_path: Path, *, num_classes: int, device: torch.device) -> nn.Module:
    """Load a PyTorch model from a state dict or pickled module."""

    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, nn.Module):
        return obj.to(device)

    state_dict = None
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj

    if state_dict is None:
        raise TypeError(f"Unsupported PyTorch artifact type: {type(obj)}")

    model = build_model(num_classes).to(device)
    model.load_state_dict(state_dict)
    return model


@app.command()
def main(
    model_path: Path = MODEL_PATH_OPTION,
    report: Path = REPORT_OPTION,
    batch_size: int = BATCH_SIZE_OPTION,
    latency_batch_size: int = LATENCY_BATCH_SIZE_OPTION,
    warmup_steps: int = WARMUP_STEPS_OPTION,
    steps: int = STEPS_OPTION,
) -> None:
    """Run a model gate (evaluation + small benchmark) and write a markdown report."""

    device = torch.device("cpu")

    test_ds = TrafficSignsDataset("test")
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)

    suffix = model_path.suffix.lower()
    model_format = suffix.lstrip(".") if suffix else "unknown"

    if suffix == ".onnx":
        import onnxruntime  # type: ignore

        session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        test_loss, test_acc = _evaluate_onnx(session=session, loader=test_loader)
        p50_ms, p95_ms = _benchmark_latency_ms_onnx(
            session,
            batch_size=latency_batch_size,
            warmup_steps=warmup_steps,
            steps=steps,
        )
    else:
        num_classes = int(torch.unique(test_ds.targets).numel())
        model = _load_torch_model_from_file(model_path, num_classes=num_classes, device=device)
        test_loss, test_acc = _evaluate(model, test_loader, device)
        p50_ms, p95_ms = _benchmark_latency_ms(
            model,
            batch_size=latency_batch_size,
            warmup_steps=warmup_steps,
            steps=steps,
            device=device,
        )

    logger.info("Test loss: {:.4f}", test_loss)
    logger.info("Test accuracy: {:.2f}%", test_acc)
    logger.info("Latency p50: {:.2f} ms (batch={})", p50_ms, latency_batch_size)
    logger.info("Latency p95: {:.2f} ms (batch={})", p95_ms, latency_batch_size)

    _write_report(
        report,
        model_path=model_path,
        model_format=model_format,
        test_loss=test_loss,
        test_acc=test_acc,
        p50_ms=p50_ms,
        p95_ms=p95_ms,
        batch_size=latency_batch_size,
        warmup_steps=warmup_steps,
        steps=steps,
    )


if __name__ == "__main__":
    app()
