"""Distributed training helpers (M30).

This module provides small, reusable utilities for running training with
PyTorch Distributed Data Parallel (DDP) via ``torchrun``.

It intentionally avoids any Hydra-specific logic so it can be imported from
different entry points.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import torch
import torch.distributed as dist
from loguru import logger


@dataclass(frozen=True)
class DistributedContext:
    """Runtime information about a (potential) distributed training run."""

    enabled: bool
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    backend: str | None

    @property
    def is_main(self) -> bool:
        """Return True for the main process (rank 0)."""

        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def distributed_env() -> tuple[int, int, int]:
    """Return (world_size, rank, local_rank) from ``torchrun`` environment."""

    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    return world_size, rank, local_rank


def init_distributed(*, enable: bool, base_device: torch.device) -> DistributedContext:
    """Initialize torch.distributed (if applicable) and return a context.

    Args:
    enable: Whether distributed training should be attempted.
    base_device: Requested device from config (used when not distributed).

    Returns:
    DistributedContext describing the run.
    """

    world_size, rank, local_rank = distributed_env()
    if not enable or world_size <= 1:
        return DistributedContext(
            enabled=False,
            world_size=1,
            rank=0,
            local_rank=0,
            device=base_device,
            backend=None,
        )

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    backend: str
    if base_device.type == "cuda" and torch.cuda.is_available() and not sys.platform.startswith("win"):
        backend = "nccl"
    else:
        backend = "gloo"

    if base_device.type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
        logger.info(
            "Initialized DDP: backend={} world_size={} rank={} local_rank={}",
            backend,
            world_size,
            rank,
            local_rank,
        )

    return DistributedContext(
        enabled=True,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    """Tear down the distributed process group (best-effort)."""

    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_loss_accuracy(
    *,
    loss_sum: float,
    correct: int,
    total: int,
    device: torch.device,
) -> tuple[float, float]:
    """All-reduce (sum) and compute global avg loss and accuracy.

    Args:
        loss_sum: Sum of per-sample losses (loss * batch_size accumulated).
        correct: Number of correct predictions.
        total: Number of samples.
        device: Device to place reduction tensor on.

    Returns:
        Tuple of (avg_loss, accuracy_percent) aggregated across all ranks.
    """

    if total <= 0:
        return 0.0, 0.0

    metrics = torch.tensor([loss_sum, float(correct), float(total)], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

    loss_sum_all, correct_all, total_all = (
        float(metrics[0].item()),
        float(metrics[1].item()),
        float(metrics[2].item()),
    )
    if total_all <= 0:
        return 0.0, 0.0
    avg_loss = loss_sum_all / total_all
    acc = 100.0 * correct_all / total_all
    return avg_loss, acc


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying model when wrapped by DDP/DataParallel."""

    return getattr(model, "module", model)
