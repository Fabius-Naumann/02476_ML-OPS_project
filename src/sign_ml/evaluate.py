
from pathlib import Path
import sys
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sign_ml.data import TrafficSignsDataset
from sign_ml.model import build_model

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def _device_from_cfg(device: str) -> torch.device:
    if device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def evaluate(model, loader, criterion, device):
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

@hydra.main(config_path="configs_files", config_name="config", version_base=None)
def main(cfg: DictConfig):
    hparams = cfg.experiment
    log = logging.getLogger(__name__)
    log.info(f"Evaluating experiment: {hparams.get('name', 'unknown')}")
    log.info(f"Hyperparameters:")
    log.info(f"  name: {hparams.get('name', '')}")
    log.info(f"  training.batch_size: {hparams.training.batch_size}")

    device = _device_from_cfg(str(cfg.device))
    batch_size = int(hparams.training.batch_size)
    
    # Get model path from config file (cfg.paths.model_out in src/sign_ml/configs_files/config.yaml)
    model_out = Path(cfg.paths.model_out)
    if not model_out.is_absolute():
        model_out = BASE_DIR / model_out

    test_ds = TrafficSignsDataset("test")
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    num_classes = len(torch.unique(test_ds.targets))
    model = build_model(num_classes).to(device)
    if not model_out.exists():
        raise FileNotFoundError(f"Model file not found at '{model_out}'. Please train the model first.")
    model.load_state_dict(torch.load(model_out, map_location=device))
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    log.info(f"Test samples: {len(test_ds)}")
    log.info(f"Test Loss: {test_loss:.4f}")
    log.info(f"Test Accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
