import datetime
import os
from pathlib import Path

import hydra
import torch
import torch.nn as nn

# Weights & Biases
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig
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

# Load environment variables once (e.g., WANDB_API_KEY, WANDB_PROJECT)
load_dotenv()


os.environ.setdefault("PROJECT_ROOT", BASE_DIR.as_posix())

now = datetime.datetime.now()

# Set up loguru to log to file in <PROJECT_ROOT>/outputs/<date>/<time>/evaluate.log
log_dir = BASE_DIR / "outputs" / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / "evaluate.log"
logger.add(str(log_file))


def evaluate(model, loader, criterion, device):
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


def evaluate_profiled(model, loader, criterion, device, *, prof, max_steps: int):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    use_non_blocking = device.type == "cuda"
    steps = 0
    with torch.inference_mode():
        for images, labels in loader:
            if steps >= max_steps:
                break
            images = images.to(device, non_blocking=use_non_blocking)
            labels = labels.to(device, non_blocking=use_non_blocking)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            prof.step()
            steps += 1
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, 100.0 * correct / total


@hydra.main(config_path=str(CONFIGS_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig):
    hparams = cfg.experiment
    logger.info("Evaluating experiment: {}", hparams.get("name", "unknown"))
    logger.info("Hyperparameters:")
    logger.info("  name: {}", hparams.get("name", ""))
    logger.info("  training.batch_size: {}", hparams.training.batch_size)

    device = device_from_cfg(str(cfg.device))
    batch_size = int(hparams.training.batch_size)

    # Get model path from config file (cfg.paths.model_out in configs/config.yaml)
    model_out = Path(cfg.paths.model_out)
    if not model_out.is_absolute():
        model_out = BASE_DIR / model_out

    test_ds = TrafficSignsDataset("test")
    pin_memory = device.type == "cuda"
    test_loader = DataLoader(test_ds, batch_size=batch_size, pin_memory=pin_memory)
    num_classes = len(torch.unique(test_ds.targets))
    model = build_model(num_classes).to(device)
    if not model_out.exists():
        raise FileNotFoundError(f"Model file not found at '{model_out}'. Please train the model first.")
    model.load_state_dict(torch.load(model_out, map_location=device))
    criterion = nn.CrossEntropyLoss()

    # Optional behavior: when enabled, create TensorBoard profiler traces under project-root ./log/.

    use_torch_profiler = _bool_from_cfg(cfg, "profiling.torch.enabled", default=False)
    export_chrome = _bool_from_cfg(cfg, "profiling.torch.export_chrome", default=False)
    export_tensorboard = _bool_from_cfg(cfg, "profiling.torch.export_tensorboard", default=True)
    max_steps = _int_from_cfg(cfg, "profiling.torch.steps", default=10)

    if use_torch_profiler:
        from torch.profiler import profile

        activities, schedule, on_trace_ready, trace_dir, tb_dir = _get_torch_profiler_config(
            cfg,
            device,
            steps=max_steps,
            timestamp=now,
            export_tensorboard=export_tensorboard,
        )

        logger.info("torch.profiler enabled: profiling {} evaluation steps", max_steps)
        with profile(
            activities=activities,
            record_shapes=True,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
        ) as prof:
            test_loss, test_acc = evaluate_profiled(
                model, test_loader, criterion, device, prof=prof, max_steps=max_steps
            )

        if export_tensorboard and tb_dir is not None:
            logger.info("torch.profiler TensorBoard logs written to: {}", tb_dir)
        if export_chrome:
            trace_path = trace_dir / "trace_eval.json"
            prof.export_chrome_trace(str(trace_path))
            logger.info("torch.profiler chrome trace written to: {}", trace_path)
    else:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    logger.info("Test samples: {}", len(test_ds))
    logger.info("Test Loss: {:.4f}", test_loss)
    logger.info("Test Accuracy: {:.2f}%", test_acc)

    # Log evaluation metrics and model artifact to wandb (fail-soft)
    use_wandb, wandb_error = init_wandb(cfg, run_name=None, group=hparams.get("name", None))
    if use_wandb:
        import wandb  # type: ignore

        wandb.log(
            {
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "test/samples": len(test_ds),
            }
        )
        artifact_name = f"model-evaluation-{hparams.get('name', 'unnamed')}-{now.strftime('%Y%m%d-%H%M%S')}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(str(model_out))
        wandb.log_artifact(artifact)
        wandb.finish()
    elif wandb_error is not None:
        logger.warning("WandB disabled during evaluation due to error: {}", wandb_error)


if __name__ == "__main__":
    main()
