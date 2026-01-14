
from pathlib import Path
import sys
from loguru import logger
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from sign_ml.data import TrafficSignsDataset
from sign_ml.model import build_model
from sign_ml.utils import device_from_cfg


BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Set up loguru to log to file in outputs/<date>/<time>/evaluate.log
now = datetime.datetime.now()
log_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "evaluate.log"
logger.add(str(log_file))

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

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs"

@hydra.main(config_path=str(CONFIG_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig):
    hparams = cfg.experiment
    logger.info("Evaluating experiment: {}", hparams.get('name', 'unknown'))
    logger.info("Hyperparameters:")
    logger.info("  name: {}", hparams.get('name', ''))
    logger.info("  training.batch_size: {}", hparams.training.batch_size)

    device = device_from_cfg(str(cfg.device))
    batch_size = int(hparams.training.batch_size)
    
    # Get model path from config file (cfg.paths.model_out in configs/config.yaml)
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
    logger.info("Test samples: {}", len(test_ds))
    logger.info("Test Loss: {:.4f}", test_loss)  
    logger.info("Test Accuracy: {:.2f}%", test_acc)

if __name__ == "__main__":
    main()
