"""Unit tests for distributed data loading and benchmarking.

Covers:
- DataLoader construction with/without DistributedSampler
- Benchmark routine behavior on normal and empty datasets
- Config loading and merge logic
"""

from __future__ import annotations

import io
from collections.abc import Iterator

import pytest
import torch
from loguru import logger
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from sign_ml.data_distributed import (
    benchmark_dataloader_loading,
    build_dataloader,
    load_experiment_cfg,
)


class TinyDataset(Dataset):
    """Simple dataset that yields a fixed number of items."""

    def __init__(self, length: int = 10) -> None:
        self.length = length
        self._data = torch.arange(length, dtype=torch.int64)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor([float(self._data[idx])], dtype=torch.float32)
        y = torch.tensor(self._data[idx], dtype=torch.int64)
        return x, y


@pytest.mark.parametrize(
    "world_size, distributed, expect_sampler",
    [
        (1, False, False),
        (1, True, False),  # distributed requested but single process → no sampler
        (2, True, True),  # multi-process distributed → sampler present
    ],
)
def test_build_dataloader_sampler_behavior(
    world_size: int, distributed: bool, expect_sampler: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify sampler presence depends on WORLD_SIZE and `distributed` flag."""

    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setenv("RANK", "0")

    ds = TinyDataset(length=8)
    loader, sampler = build_dataloader(
        ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        multiprocessing_context=None,
        distributed=distributed,
    )

    if expect_sampler:
        assert isinstance(sampler, DistributedSampler)
        # Sampler reflects world size
        assert sampler.num_replicas == world_size
    else:
        assert sampler is None

    # Basic loader sanity: produces items
    it = iter(loader)
    batch = next(it)
    assert isinstance(batch, (list, tuple))
    assert len(batch) == 2
    assert isinstance(batch[0], torch.Tensor)
    assert isinstance(batch[1], torch.Tensor)


def test_benchmark_on_small_dataset_logs_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Benchmark should log metrics and not raise on small datasets."""

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    ds = TinyDataset(length=10)
    loader, sampler = build_dataloader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        multiprocessing_context=None,
        distributed=False,
    )

    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO")
    try:
        benchmark_dataloader_loading(loader, sampler, batches_to_check=3)
    finally:
        logger.remove(sink_id)

    output = sink.getvalue()
    assert "Benchmark:" in output
    assert "imgs/s" in output


def test_benchmark_on_empty_dataset_warns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Benchmark should warn and return when no batches are produced."""

    class EmptyDataset(Dataset):
        def __len__(self) -> int:  # pragma: no cover - trivial
            return 0

        def __getitem__(self, idx: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:  # pragma: no cover
            raise IndexError

    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    ds = EmptyDataset()
    loader, sampler = build_dataloader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        multiprocessing_context=None,
        distributed=False,
    )

    sink = io.StringIO()
    sink_id = logger.add(sink, level="INFO")
    try:
        benchmark_dataloader_loading(loader, sampler, batches_to_check=2)
    finally:
        logger.remove(sink_id)

    output = sink.getvalue()
    assert "No batches produced" in output


def test_load_experiment_cfg_defaults() -> None:
    """Defaults from base config should select an experiment and merge content."""

    cfg = load_experiment_cfg(experiment_name=None)
    assert hasattr(cfg, "experiment")
    # Expect common fields present
    assert hasattr(cfg.experiment, "training")
    assert hasattr(cfg.experiment.training, "batch_size")


def test_load_experiment_cfg_missing_raises() -> None:
    """Missing experiment name should raise with available list in message."""

    with pytest.raises(FileNotFoundError):
        load_experiment_cfg(experiment_name="__nonexistent_experiment__")
