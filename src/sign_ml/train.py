from pathlib import Path

import contextlib
from loguru import logger
import random

import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Weights & Biases
import wandb
from dotenv import load_dotenv
import os

# Allow running this file directly (e.g. `python src/sign_ml/train.py`) while keeping
# package-correct imports for VS Code navigation.
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sign_ml.data import TrafficSignsDataset
from sign_ml.model import build_model
from sign_ml.utils import device_from_cfg

BASE_DIR = Path(__file__).resolve().parent.parent.parent

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

import datetime

# Set up loguru to log to file in outputs/<date>/<time>/train.log
now = datetime.datetime.now()
log_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "train.log"
logger.add(str(log_file))


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducible training."""

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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
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

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total





def train(cfg: DictConfig) -> Path:
    """Train the traffic sign model using a Hydra configuration."""

    # Load environment variables (for WANDB_API_KEY, etc.)
    load_dotenv()

    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))

    hparams = cfg.experiment

    seed = int(hparams.seed)
    _set_seed(seed)

    # Initialize wandb (fail-soft if not permitted)
    use_wandb = False
    wandb_project = os.getenv("WANDB_PROJECT", "sign-ml")
    wandb_entity = os.getenv("WANDB_ENTITY")
    try:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            name=hparams.get("name", None),
        )
        use_wandb = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("WandB disabled due to error: {}", exc)
    device = device_from_cfg(str(cfg.device))

    train_ds = TrafficSignsDataset("train")
    val_ds = TrafficSignsDataset("val")

    generator = torch.Generator()
    generator.manual_seed(seed)

    batch_size = int(hparams.training.batch_size)
    epochs = int(hparams.training.epochs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, generator=generator)

    num_classes = len(torch.unique(train_ds.targets))
    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, lr=float(hparams.optimizer.lr), params=model.parameters())

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
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
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(str(model_out))
        wandb.log_artifact(artifact)
        wandb.finish()
    return model_out


@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""

    train(cfg)


if __name__ == "__main__":
    main()
