"""Distributed-aware data loading and benchmarking (M29).

This module isolates the distributed DataLoader construction and throughput benchmarking to keep
``data.py`` focused on preprocessing and visualization.

Usage:
- Programmatic (preferred): ``from sign_ml.data import benchmark_loading_from_config``
- Direct import (advanced): ``from sign_ml.data_distributed import benchmark_loading_from_config``

Note: The CLI was removed. Use the programmatic entry points instead.
"""

from __future__ import annotations

import os
from time import perf_counter
from typing import Any, cast

import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from sign_ml import CONFIGS_DIR
from sign_ml.data import TrafficSignsDataset


def _distributed_env() -> tuple[int, int]:
    """Get (world_size, rank) from environment (torchrun variables)."""

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    return world_size, rank


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader workers for determinism."""

    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed + worker_id)


def build_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool,
    pin_memory: bool,
    multiprocessing_context: str | None,
    distributed: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    """Create a DataLoader with optional distributed sharding."""

    world_size, rank = _distributed_env()

    sampler: DistributedSampler | None = None
    effective_shuffle = shuffle
    if distributed and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=False,
        )
        effective_shuffle = False

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": effective_shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker if num_workers > 0 else None,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers
        if multiprocessing_context is not None:
            loader_kwargs["multiprocessing_context"] = multiprocessing_context

    loader = DataLoader(dataset, **cast(Any, loader_kwargs))
    return loader, sampler


def benchmark_dataloader_loading(
    dataloader: DataLoader,
    sampler: DistributedSampler | None,
    *,
    batches_to_check: int,
) -> None:
    """Benchmark how quickly batches can be produced by a DataLoader."""

    if sampler is not None:
        sampler.set_epoch(0)

    warmup_batches = 1
    batch_times: list[float] = []
    images_total = 0

    data_iter = iter(dataloader)
    for _ in range(warmup_batches):
        try:
            next(data_iter)
        except StopIteration:
            logger.warning("No batches produced; nothing to benchmark.")
            return

    start = perf_counter()
    for _ in range(batches_to_check):
        try:
            images, _labels = next(data_iter)
        except StopIteration:
            break
        end = perf_counter()
        batch_times.append(end - start)
        start = end
        images_total += int(images.shape[0])

    if not batch_times:
        logger.warning("No batches produced; nothing to benchmark.")
        return

    total_s = sum(batch_times)
    avg_s = total_s / len(batch_times)
    imgs_per_s = images_total / total_s if total_s > 0 else float("inf")
    s_per_img = total_s / images_total if images_total > 0 else float("inf")
    logger.info(
        "Benchmark: warmup_batches={} batches={} images={} total={:.3f}s avg_batch={:.4f}s throughput={:.1f} imgs/s "
        "({:.6f}s/img = {:.3f}ms/img)",
        warmup_batches,
        len(batch_times),
        images_total,
        total_s,
        avg_s,
        imgs_per_s,
        s_per_img,
        s_per_img * 1000.0,
    )


def load_experiment_cfg(*, experiment_name: str | None) -> Any:
    """Load the selected experiment config (base + experiment)."""

    from omegaconf import DictConfig, OmegaConf

    base_cfg = OmegaConf.load(CONFIGS_DIR / "config.yaml")
    if not isinstance(base_cfg, DictConfig):
        raise TypeError(f"Expected configs/config.yaml to load as DictConfig, got {type(base_cfg)!r}")

    if experiment_name is None:
        defaults_any = base_cfg.get("defaults", [])
        defaults: list[Any] = list(defaults_any) if defaults_any is not None else []
        selected: str | None = None
        for item in defaults:
            if OmegaConf.is_dict(item) and item.get("experiment") is not None:
                selected = str(item.get("experiment"))
                break
        experiment_name = selected

    if experiment_name is None:
        raise RuntimeError(
            "No experiment selected. Set defaults.experiment in configs/config.yaml or pass an experiment name."
        )

    exp_cfg_path = CONFIGS_DIR / "experiment" / f"{experiment_name}.yaml"
    if not exp_cfg_path.is_file():
        experiments_dir = CONFIGS_DIR / "experiment"
        available = sorted(p.stem for p in experiments_dir.glob("*.yaml"))
        if available:
            raise FileNotFoundError(
                f"Experiment config file for '{experiment_name}' not found at {exp_cfg_path}. "
                f"Available experiments: {', '.join(available)}"
            )
        raise FileNotFoundError(
            f"Experiment config file for '{experiment_name}' not found at {exp_cfg_path}. "
            f"No experiment configs were found in {experiments_dir}."
        )
    exp_cfg = OmegaConf.load(exp_cfg_path)
    return OmegaConf.merge(base_cfg, {"experiment": exp_cfg})


def benchmark_loading_from_config(
    *,
    experiment: str | None = None,
    split: str = "train",
    distributed: bool = False,
    batch_size: int | None = None,
    num_workers: int | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool | None = None,
    pin_memory: bool | None = None,
    multiprocessing_context: str | None = None,
    batches_to_check: int | None = None,
) -> None:
    """Run the DataLoader benchmark using config defaults (no CLI required)."""

    cfg = load_experiment_cfg(experiment_name=experiment)
    cfg_batch_size = int(cfg.experiment.training.batch_size)

    dl_cfg = cfg.experiment.get("data_loading", {})
    cfg_num_workers = int(dl_cfg.get("num_workers", 0))
    cfg_prefetch_factor = int(dl_cfg.get("prefetch_factor", 2))
    cfg_persistent_workers = bool(dl_cfg.get("persistent_workers", False))
    cfg_pin_memory = dl_cfg.get("pin_memory", None)
    cfg_multiprocessing_context = dl_cfg.get("multiprocessing_context", None)
    cfg_batches_to_check = int(dl_cfg.get("batches_to_check", 64))

    batch_size = cfg_batch_size if batch_size is None else int(batch_size)
    num_workers = cfg_num_workers if num_workers is None else int(num_workers)
    prefetch_factor = cfg_prefetch_factor if prefetch_factor is None else int(prefetch_factor)
    persistent_workers = cfg_persistent_workers if persistent_workers is None else bool(persistent_workers)

    if multiprocessing_context is None and cfg_multiprocessing_context is not None:
        multiprocessing_context = str(cfg_multiprocessing_context)

    if pin_memory is None:
        pin_memory = bool(torch.cuda.is_available()) if cfg_pin_memory is None else bool(cfg_pin_memory)
    else:
        pin_memory = bool(pin_memory)

    batches_to_check = cfg_batches_to_check if batches_to_check is None else int(batches_to_check)

    split_lower = split.lower()
    if split_lower not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")

    ds = TrafficSignsDataset(split_lower)
    shuffle = split_lower == "train"
    loader, sampler = build_dataloader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        multiprocessing_context=multiprocessing_context,
        distributed=distributed,
    )
    world_size, rank = _distributed_env()
    logger.info(
        "DataLoader config: split={} batch_size={} num_workers={} distributed={} world_size={} rank={}",
        split_lower,
        batch_size,
        num_workers,
        distributed,
        world_size,
        rank,
    )
    benchmark_dataloader_loading(loader, sampler, batches_to_check=batches_to_check)


## Note: CLI was intentionally removed as it's not needed. Use
## sign_ml.data.benchmark_loading_from_config for programmatic access.
