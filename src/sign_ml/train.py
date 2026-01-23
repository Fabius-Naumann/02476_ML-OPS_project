import contextlib
import datetime
import os
import random
from pathlib import Path
from typing import Any

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
from torch.utils.data.distributed import DistributedSampler

from sign_ml import BASE_DIR, CONFIGS_DIR
from sign_ml.data import TrafficSignsDataset
from sign_ml.Distributed_Training import cleanup_distributed, init_distributed, reduce_loss_accuracy, unwrap_model
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

    for step_idx, (images, labels) in enumerate(loader):
        if step_idx >= max_steps:
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

    return reduce_loss_accuracy(loss_sum=total_loss, correct=correct, total=total, device=device)


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

    return reduce_loss_accuracy(loss_sum=total_loss, correct=correct, total=total, device=device)


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

    return reduce_loss_accuracy(loss_sum=total_loss, correct=correct, total=total, device=device)


def _run_profiled_epoch_zero(
    *,
    cfg: DictConfig,
    device: torch.device,
    torch_profiler_steps: int,
    run_timestamp: datetime.datetime,
    export_tensorboard: bool,
    export_chrome: bool,
    model,
    train_loader,
    criterion,
    optimizer,
) -> tuple[float, float]:
    """Run a limited, profiled training epoch (epoch 0).

    Returns the training loss and accuracy computed over ``torch_profiler_steps`` iterations.
    Also handles exporting TensorBoard traces and Chrome JSON traces if enabled in config.
    """

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

    return train_loss, train_acc


def _setup_distributed(cfg: DictConfig) -> tuple[Any, torch.device]:
    """Initialize (optional) distributed training and return (context, device)."""

    distributed_cfg = _bool_from_cfg(cfg, "distributed.enabled", default=False)
    distributed_env = int(os.getenv("WORLD_SIZE", "1")) > 1
    distributed_enabled = distributed_cfg or distributed_env

    base_device = device_from_cfg(str(cfg.device))
    dist_ctx = init_distributed(enable=distributed_enabled, base_device=base_device)
    return dist_ctx, dist_ctx.device


def _setup_logging(*, dist_ctx: Any, run_timestamp: datetime.datetime) -> Path:
    """Set up output directory and file logging (rank-safe)."""

    date_str = run_timestamp.strftime("%Y-%m-%d")
    time_str = run_timestamp.strftime("%H-%M-%S")
    log_suffix = f"-rank{dist_ctx.rank}" if dist_ctx.enabled else ""
    log_dir = Path("outputs") / date_str / f"{time_str}{log_suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_dir / f"train{log_suffix}.log"))
    return log_dir


def _init_wandb_if_main(*, cfg: DictConfig, dist_ctx: Any, hparams: Any) -> bool:
    """Initialize W&B only on the main rank (rank 0)."""

    if not dist_ctx.is_main:
        return False

    use_wandb, wandb_error = init_wandb(cfg, run_name=None, group=hparams.get("name", None))
    if not use_wandb and wandb_error is not None:
        logger.warning("WandB disabled due to error: {}", wandb_error)

    if use_wandb:
        import wandb  # type: ignore

        is_sweep_run = getattr(getattr(wandb, "run", None), "sweep_id", None) is not None or os.getenv("WANDB_SWEEP_ID")
        if is_sweep_run:
            logger.info("W&B sweep objective: validation/accuracy (maximize)")

    return use_wandb


def _build_loaders(
    *,
    train_ds: TrafficSignsDataset,
    val_ds: TrafficSignsDataset,
    batch_size: int,
    generator: torch.Generator,
    device: torch.device,
    dist_ctx: Any,
    seed: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """Build train/val DataLoaders and an optional DistributedSampler for training."""

    pin_memory = device.type == "cuda"

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
            seed=seed,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        generator=generator,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        generator=generator,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_sampler


def _maybe_wrap_ddp(*, model: torch.nn.Module, device: torch.device, dist_ctx: Any) -> torch.nn.Module:
    """Wrap model in DistributedDataParallel when enabled."""

    if not dist_ctx.enabled:
        return model

    from torch.nn.parallel import DistributedDataParallel as DDP

    if device.type == "cuda":
        return DDP(model, device_ids=[dist_ctx.local_rank], output_device=dist_ctx.local_rank)
    return DDP(model)


def _train_epochs(
    *,
    cfg: DictConfig,
    hparams: Any,
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: DistributedSampler | None,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    dist_ctx: Any,
    use_wandb: bool,
    run_timestamp: datetime.datetime,
) -> None:
    """Run the training loop (rank-aware logging and W&B)."""

    use_torch_profiler = _bool_from_cfg(cfg, "profiling.torch.enabled", default=False)
    torch_profiler_steps = _int_from_cfg(cfg, "profiling.torch.steps", default=10)
    export_chrome = _bool_from_cfg(cfg, "profiling.torch.export_chrome", default=True)
    export_tensorboard = _bool_from_cfg(cfg, "profiling.torch.export_tensorboard", default=False)

    if dist_ctx.enabled and use_torch_profiler:
        logger.info("Disabling torch.profiler because distributed training is enabled.")
        use_torch_profiler = False

    epochs = int(hparams.training.epochs)
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        if use_torch_profiler and epoch == 0:
            train_loss, train_acc = _run_profiled_epoch_zero(
                cfg=cfg,
                device=device,
                torch_profiler_steps=torch_profiler_steps,
                run_timestamp=run_timestamp,
                export_tensorboard=export_tensorboard,
                export_chrome=export_chrome,
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
            )
            val_loss, val_acc = float("nan"), float("nan")
            logger.info(
                "Skipping validation for profiled epoch 0 because training ran on a limited number of steps "
                "(torch_profiler_steps={}).",
                torch_profiler_steps,
            )
        else:
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = validate(model, val_loader, criterion, device)

        if dist_ctx.is_main:
            logger.info(
                "Epoch [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.2f}% | Val Loss: {:.4f} | Val Acc: {:.2f}%",
                epoch + 1,
                epochs,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
            )

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


def _save_model_if_main(*, model: torch.nn.Module, cfg: DictConfig, dist_ctx: Any) -> Path:
    """Save model state dict (rank 0 only) and return the output path."""

    model_out = _resolve_path(BASE_DIR, str(cfg.paths.model_out))
    if dist_ctx.is_main:
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(unwrap_model(model).state_dict(), model_out)
        logger.info("Model saved to: {}", model_out)
    return model_out


def train(cfg: DictConfig) -> Path:
    """Train the traffic sign model using a Hydra configuration."""

    dist_ctx, device = _setup_distributed(cfg)
    run_timestamp = datetime.datetime.now()
    _setup_logging(dist_ctx=dist_ctx, run_timestamp=run_timestamp)
    logger.info("Configuration:\n{}", OmegaConf.to_yaml(cfg))

    hparams = cfg.experiment
    seed = int(hparams.seed)
    _set_seed(seed + dist_ctx.rank)

    use_wandb = _init_wandb_if_main(cfg=cfg, dist_ctx=dist_ctx, hparams=hparams)

    train_ds = TrafficSignsDataset("train")
    val_ds = TrafficSignsDataset("val")
    generator = torch.Generator().manual_seed(seed)

    batch_size = int(hparams.training.batch_size)
    train_loader, val_loader, train_sampler = _build_loaders(
        train_ds=train_ds,
        val_ds=val_ds,
        batch_size=batch_size,
        generator=generator,
        device=device,
        dist_ctx=dist_ctx,
        seed=seed,
    )

    num_classes = len(torch.unique(train_ds.targets))
    model = _maybe_wrap_ddp(model=build_model(num_classes).to(device), device=device, dist_ctx=dist_ctx)

    criterion = nn.CrossEntropyLoss()
    optimizer = instantiate(cfg.optimizer, lr=float(hparams.optimizer.lr), params=model.parameters())

    try:
        _train_epochs(
            cfg=cfg,
            hparams=hparams,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            dist_ctx=dist_ctx,
            use_wandb=use_wandb,
            run_timestamp=run_timestamp,
        )

        model_out = _save_model_if_main(model=model, cfg=cfg, dist_ctx=dist_ctx)
        if dist_ctx.is_main and use_wandb:
            import wandb  # type: ignore

            artifact_name = f"model-train-{hparams.get('name', 'unnamed')}-{run_timestamp.strftime('%Y%m%d-%H%M%S')}"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.add_file(str(model_out))
            wandb.log_artifact(artifact)
            wandb.finish()

        return model_out
    finally:
        cleanup_distributed(dist_ctx)


@hydra.main(config_path=str(CONFIGS_DIR), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra entry point for training."""

    train(cfg)


if __name__ == "__main__":
    main()
