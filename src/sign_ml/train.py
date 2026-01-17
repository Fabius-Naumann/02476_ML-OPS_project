import contextlib
import datetime
import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn as nn

# Weights & Biases
from dotenv import load_dotenv
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from sign_ml import BASE_DIR, CONFIGS_DIR
from sign_ml.data import TrafficSignsDataset
from sign_ml.model import build_model
from sign_ml.utils import (
    _bool_from_cfg,
    _get_torch_profiler_config,
    _int_from_cfg,
    device_from_cfg,
    init_wandb,
)


os.environ.setdefault("PROJECT_ROOT", BASE_DIR.as_posix())


# Load environment variables once (e.g., WANDB_API_KEY, WANDB_PROJECT)
load_dotenv()


def train_one_epoch_profiled(
    model,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    *,
    prof,
    max_steps: int,
):
    """Train for a limited number of steps while calling `prof.step()` each iteration."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_non_blocking = device.type == "cuda"

    steps = 0
    for images, labels in loader:
        if steps >= max_steps:
            break

        images = images.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        prof.step()
        steps += 1

    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with contextlib.suppress(Exception):
        torch.use_deterministic_algorithms(True)


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else base_dir / path


def train_one_epoch(model, loader, criterion, optimizer, device: torch.device):
    """Train the model for one epoch."""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    use_non_blocking = device.type == "cuda"

    for images, labels in loader:
        images = images.to(device, non_blocking=use_non_blocking)
        labels = labels.to(device, non_blocking=use_non_blocking)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def validate(model, loader, criterion, device: torch.device):
    """Evaluate the model on a validation set."""

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    use_non_blocking = device.type == "cuda"

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def train(cfg: DictConfig) -> Path:
    """Train the traffic sign model using a Hydra configuration."""

    run_timestamp = datetime.datetime.now()
    logger.add("train.log")

    date_str = run_timestamp.strftime("%Y-%m-%d")
    time_str = run_timestamp.strftime("%H-%M-%S")
    log_dir = Path("outputs") / date_str / time_str
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / "train.log"))

    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))

    hparams = cfg.experiment

    seed = int(hparams.seed)
    _set_seed(seed)

    # Initialize wandb (fail-soft if not permitted)
    use_wandb, wandb_error = init_wandb(cfg, run_name=None, group=hparams.get("name", None))
    if not use_wandb and wandb_error is not None:
        logger.warning("WandB disabled due to error: {}", wandb_error)

    if use_wandb:
        import wandb  # type: ignore

        is_sweep_run = getattr(getattr(wandb, "run", None), "sweep_id", None) is not None or os.getenv("WANDB_SWEEP_ID")
        if is_sweep_run:
            logger.info("W&B sweep objective: validation/accuracy (maximize)")
    device = device_from_cfg(str(cfg.device))

    train_ds = TrafficSignsDataset("train")
    val_ds = TrafficSignsDataset("val")

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(hparams.training.batch_size)
    epochs = int(hparams.training.epochs)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        generator=generator,
        pin_memory=pin_memory,
    )

    num_classes = len(torch.unique(train_ds.targets))
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, lr=float(hparams.optimizer.lr), params=model.parameters())

    use_torch_profiler = _bool_from_cfg(cfg, "profiling.torch.enabled", default=False)
    torch_profiler_steps = _int_from_cfg(cfg, "profiling.torch.steps", default=10)
    export_chrome = _bool_from_cfg(cfg, "profiling.torch.export_chrome", default=True)
    export_tensorboard = _bool_from_cfg(cfg, "profiling.torch.export_tensorboard", default=False)

    for epoch in range(epochs):
        if use_torch_profiler and epoch == 0:
            from torch.profiler import profile

            activities, schedule, on_trace_ready, trace_dir, tb_dir = _get_torch_profiler_config(
                cfg,
                device,
                steps=torch_profiler_steps,
                timestamp=run_timestamp,
                export_tensorboard=export_tensorboard,
            )

            logger.info("torch.profiler enabled: profiling {} training steps", torch_profiler_steps)
            with profile(
                activities=activities,
                record_shapes=True,
                schedule=schedule,
                on_trace_ready=on_trace_ready,
            ) as prof:
                train_loss, train_acc = train_one_epoch_profiled(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    device,
                    prof=prof,
                    max_steps=torch_profiler_steps,
                )

            if export_tensorboard and tb_dir is not None:
                logger.info("torch.profiler TensorBoard logs written to: {}", tb_dir)

            if export_chrome:
                trace_path = trace_dir / "trace.json"
                prof.export_chrome_trace(str(trace_path))
                logger.info("torch.profiler chrome trace written to: {}", trace_path)
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        if use_torch_profiler and epoch == 0:
            val_loss, val_acc = float("nan"), float("nan")
            logger.info(
                "Skipping validation for profiled epoch 0 because training ran on a limited number of steps "
                "(torch_profiler_steps={}).",
                torch_profiler_steps,
            )
        else:
            val_loss, val_acc = validate(model, val_loader, criterion, device)

        logger.info(
            "Epoch [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.2f}% | Val Loss: {:.4f} | Val Acc: {:.2f}%",
            epoch + 1,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        # Log metrics to wandb (if enabled)
        if use_wandb:
            import wandb  # type: ignore

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                }
            )

    model_out = _resolve_path(BASE_DIR, str(cfg.paths.model_out))
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    logger.info("Model saved to: {}", model_out)
    # Log model artifact to wandb (if enabled)
    if use_wandb:
        import wandb  # type: ignore

        artifact_name = f"model-train-{hparams.get('name', 'unnamed')}-{run_timestamp.strftime('%Y%m%d-%H%M%S')}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(model_out))
        wandb.log_artifact(artifact)
        wandb.finish()
    return model_out


@hydra.main(config_path=str(CONFIGS_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""

    train(cfg)


if __name__ == "__main__":
    main()
