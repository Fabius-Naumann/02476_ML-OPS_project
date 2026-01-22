import os
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
import typer
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from sign_ml import CONFIGS_DIR, FIGURES_DIR, PROCESSED_DIR, RAW_DIR

ZIP_PATH = RAW_DIR / "traffic_signs_merged.zip"
EXTRACT_ROOT = RAW_DIR / "traffic_signs"

TRAIN_FILE = PROCESSED_DIR / "train_preprocessed.pt"
VAL_FILE = PROCESSED_DIR / "val_preprocessed.pt"
TEST_FILE = PROCESSED_DIR / "test_preprocessed.pt"

IMAGE_SIZE = (64, 64)
VAL_SPLIT = 0.2
RANDOM_SEED = 42

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def _stratified_train_val_split_indices(
    labels: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a stratified train/val split.

    This is a lightweight replacement for ``sklearn.model_selection.train_test_split(..., stratify=labels)``.

    Args:
        labels: 1D tensor with class labels.
        val_fraction: Fraction of samples to place in the validation split.
        seed: Random seed for deterministic splitting.

    Returns:
        Tuple of (train_indices, val_indices) as int64 tensors.
    """

    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction must be in (0, 1)")

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_indices: list[int] = []
    val_indices: list[int] = []

    unique_labels = torch.unique(labels)
    for label in unique_labels.tolist():
        class_indices = torch.where(labels == label)[0]
        if class_indices.numel() <= 1:
            train_indices.extend(class_indices.tolist())
            continue

        perm = torch.randperm(class_indices.numel(), generator=generator)
        shuffled = class_indices[perm]

        val_count = int(class_indices.numel() * val_fraction)
        val_count = max(1, val_count)
        val_count = min(val_count, int(class_indices.numel() - 1))

        val_indices.extend(shuffled[:val_count].tolist())
        train_indices.extend(shuffled[val_count:].tolist())

    train_tensor = torch.tensor(train_indices, dtype=torch.int64)
    val_tensor = torch.tensor(val_indices, dtype=torch.int64)

    train_tensor = train_tensor[torch.randperm(train_tensor.numel(), generator=generator)]
    val_tensor = val_tensor[torch.randperm(val_tensor.numel(), generator=generator)]
    return train_tensor, val_tensor


class NumericImageFolder(datasets.ImageFolder):
    """ImageFolder that sorts class folders numerically when possible."""

    @staticmethod
    def find_classes(directory: str) -> tuple[list[str], dict[str, int]]:
        classes = [path.name for path in Path(directory).iterdir() if path.is_dir()]
        classes.sort(key=lambda name: int(name) if name.isdigit() else name)
        class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        return classes, class_to_idx


def preprocess_data() -> None:
    """Preprocess the raw dataset and store tensors to disk."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not EXTRACT_ROOT.exists():
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_ROOT)

    DATA_DIR = EXTRACT_ROOT / "DATA"
    TEST_DIR = EXTRACT_ROOT / "TEST"

    if not DATA_DIR.exists():
        DATA_DIR = EXTRACT_ROOT / "traffic_signs" / "DATA"
        TEST_DIR = EXTRACT_ROOT / "traffic_signs" / "TEST"

    if not (DATA_DIR.exists() and TEST_DIR.exists()):
        raise FileNotFoundError(
            f"Expected traffic sign data directories not found.\n"
            f"Looked for either:\n"
            f"  - {EXTRACT_ROOT / 'DATA'} and {EXTRACT_ROOT / 'TEST'}\n"
            f"  - {EXTRACT_ROOT / 'traffic_signs' / 'DATA'} and "
            f"{EXTRACT_ROOT / 'traffic_signs' / 'TEST'}"
        )
    full_train_ds = NumericImageFolder(DATA_DIR, transform=PREPROCESS)
    test_ds = NumericImageFolder(TEST_DIR, transform=PREPROCESS)

    def to_tensors(dataset: Iterable[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        images: list[torch.Tensor] = []
        labels: list[int] = []
        for img, label in dataset:
            images.append(img)
            labels.append(label)
        return torch.stack(images), torch.tensor(labels)

    train_images, train_labels = to_tensors(full_train_ds)
    test_images, test_labels = to_tensors(test_ds)

    train_idx, val_idx = _stratified_train_val_split_indices(
        train_labels,
        val_fraction=VAL_SPLIT,
        seed=RANDOM_SEED,
    )

    torch.save({"images": train_images[train_idx], "labels": train_labels[train_idx]}, TRAIN_FILE)

    torch.save({"images": train_images[val_idx], "labels": train_labels[val_idx]}, VAL_FILE)

    torch.save({"images": test_images, "labels": test_labels}, TEST_FILE)


class TrafficSignsDataset(Dataset):
    """Dataset backed by preprocessed tensors on disk."""

    def __init__(self, split: str = "train") -> None:
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        file_path = {
            "train": str(TRAIN_FILE),
            "val": str(VAL_FILE),
            "test": str(TEST_FILE),
        }[split]

        if not Path(file_path).exists():
            preprocess_data()

        data = torch.load(file_path, weights_only=True)
        self.images = data["images"]
        self.targets = data["labels"]

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.targets[idx]


def _distributed_env() -> tuple[int, int]:
    """Get (world_size, rank) from environment.

    Uses the standard ``torchrun`` environment variables.
    """

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    return world_size, rank


def _seed_worker(worker_id: int) -> None:
    """Seed dataloader workers for determinism."""

    # torch.initial_seed() is different for each worker/process
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
    """Create a DataLoader with optional distributed sharding.

    Args:
        dataset: Dataset instance.
        batch_size: Batch size.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes.
        prefetch_factor: Prefetch batches per worker (only used when ``num_workers > 0``).
        persistent_workers: Keep worker processes alive between epochs.
        pin_memory: Enable pinned-memory transfers.
        multiprocessing_context: Multiprocessing start method (e.g., "spawn").
        distributed: If True and ``WORLD_SIZE > 1``, uses ``DistributedSampler``.

    Returns:
        Tuple of (DataLoader, sampler_or_none).
    """

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

    loader_kwargs: dict[str, object] = {
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

    loader = DataLoader(dataset, **loader_kwargs)
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

    batch_times: list[float] = []
    images_total = 0

    start = perf_counter()
    for batch_idx, (images, _labels) in enumerate(dataloader):
        if batch_idx >= batches_to_check:
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
        "Benchmark: batches={} images={} total={:.3f}s avg_batch={:.4f}s throughput={:.1f} imgs/s "
        "({:.6f}s/img = {:.3f}ms/img)",
        len(batch_times),
        images_total,
        total_s,
        avg_s,
        imgs_per_s,
        s_per_img,
        s_per_img * 1000.0,
    )


def load_experiment_cfg(*, experiment_name: str | None) -> Any:
    """Load the experiment config selected by configs/config.yaml.

    This is a lightweight alternative to invoking Hydra, but still uses the same
    config selection mechanism as train/evaluate.

    Args:
        experiment_name: Experiment name (e.g., "exp1"). If None, uses
            defaults.experiment from configs/config.yaml.

    Returns:
        An OmegaConf configuration object with the selected experiment merged in.
    """

    from omegaconf import OmegaConf

    base_cfg = OmegaConf.load(CONFIGS_DIR / "config.yaml")

    if experiment_name is None:
        defaults = base_cfg.get("defaults", [])
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
    """Run the DataLoader benchmark using config defaults (no CLI required).

    This loads configs/config.yaml + configs/experiment/<experiment>.yaml and uses
    experiment.training.batch_size + experiment.data_loading.* as defaults.
    Any function arguments override the config.

    Args:
        experiment: Experiment name (e.g., "exp1"). If None, uses
            defaults.experiment from configs/config.yaml.
        split: Dataset split (train/val/test).
        distributed: Use DistributedSampler when WORLD_SIZE>1.
        batch_size: Batch size override.
        num_workers: DataLoader workers override.
        prefetch_factor: Prefetch factor override (when workers > 0).
        persistent_workers: Keep workers alive between epochs override.
        pin_memory: Enable pinned-memory transfers override.
        multiprocessing_context: Multiprocessing context override (e.g., "spawn").
        batches_to_check: Number of batches to load during benchmarking override.
    """

    cfg = load_experiment_cfg(experiment_name=experiment)
    cfg_batch_size = int(cfg.experiment.training.batch_size)

    dl_cfg = cfg.experiment.get("data_loading", {})
    cfg_num_workers = int(dl_cfg.get("num_workers", 0))
    cfg_prefetch_factor = int(dl_cfg.get("prefetch_factor", 2))
    cfg_persistent_workers = bool(dl_cfg.get("persistent_workers", False))
    cfg_pin_memory = dl_cfg.get("pin_memory", None)
    cfg_multiprocessing_context = dl_cfg.get("multiprocessing_context", None)
    cfg_batches_to_check = int(dl_cfg.get("batches_to_check", 50))

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


# CLI entry point only available if run as a script
if __name__ == "__main__":
    BENCHMARK_OPTION = typer.Option(False, "--benchmark-loading", "-get_timing")
    EXPERIMENT_OPTION = typer.Option(None, "--experiment")
    SPLIT_OPTION = typer.Option("train", "--split")
    BATCH_SIZE_OPTION = typer.Option(None, "--batch-size", min=1)
    NUM_WORKERS_OPTION = typer.Option(None, "--num-workers", min=0)
    PREFETCH_FACTOR_OPTION = typer.Option(None, "--prefetch-factor", min=1)
    PERSISTENT_WORKERS_OPTION = typer.Option(None, "--persistent-workers")
    MULTIPROCESSING_CONTEXT_OPTION = typer.Option(None, "--multiprocessing-context")
    DISTRIBUTED_OPTION = typer.Option(False, "--distributed")
    BATCHES_TO_CHECK_OPTION = typer.Option(None, "--batches-to-check", min=1)

    def _format_class_table(split: str, targets: torch.Tensor) -> str:
        """Format class distribution statistics for a dataset split."""

        classes, counts = torch.unique(targets, return_counts=True)
        total = int(counts.sum().item())
        sorted_pairs = sorted(zip(classes.tolist(), counts.tolist(), strict=False), key=lambda pair: pair[0])

        header = f"{split} split class distribution"
        lines = [header, "Class | Count | Percent", "----- | ----- | -------"]
        for class_id, count in sorted_pairs:
            percent = (count / total) * 100.0 if total > 0 else 0.0
            lines.append(f"{class_id:>5} | {count:>5} | {percent:>6.2f}%")
        lines.append(f"Total | {total:>5} | 100.00%")
        return "\n".join(lines)

    def main(
        benchmark_loading: bool = BENCHMARK_OPTION,
        experiment: str | None = EXPERIMENT_OPTION,
        split: str = SPLIT_OPTION,
        batch_size: int | None = BATCH_SIZE_OPTION,
        num_workers: int | None = NUM_WORKERS_OPTION,
        prefetch_factor: int | None = PREFETCH_FACTOR_OPTION,
        persistent_workers: bool | None = PERSISTENT_WORKERS_OPTION,
        multiprocessing_context: str | None = MULTIPROCESSING_CONTEXT_OPTION,
        distributed: bool = DISTRIBUTED_OPTION,
        batches_to_check: int | None = BATCHES_TO_CHECK_OPTION,
    ) -> None:
        """Benchmark DataLoader performance or visualize samples.

        Args:
            benchmark_loading: Whether to benchmark DataLoader performance on the chosen split.
            experiment: Which experiment config to use (e.g., exp1). If omitted, uses configs/config.yaml defaults.
            split: Dataset split (train/val/test).
            batch_size: Batch size for benchmarking.
            num_workers: DataLoader workers for benchmarking.
            prefetch_factor: Prefetch factor for benchmarking (when workers > 0).
            persistent_workers: Keep workers alive between epochs.
            multiprocessing_context: Multiprocessing context (e.g., "spawn").
            distributed: Use DistributedSampler when WORLD_SIZE>1.
            batches_to_check: Number of batches to load during benchmarking.
        """

        if benchmark_loading:
            try:
                benchmark_loading_from_config(
                    experiment=experiment,
                    split=split,
                    distributed=distributed,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    prefetch_factor=prefetch_factor,
                    persistent_workers=persistent_workers,
                    multiprocessing_context=multiprocessing_context,
                    batches_to_check=batches_to_check,
                )
            except ValueError as exc:
                raise typer.BadParameter(str(exc)) from exc
            return

        from sign_ml.visualize import plot_samples

        train_ds = TrafficSignsDataset("train")
        val_ds = TrafficSignsDataset("val")
        test_ds = TrafficSignsDataset("test")

        # plot samples
        output = FIGURES_DIR / "samples.png"
        samples = 9
        train_output = output.with_stem(output.stem + "_train")
        val_output = output.with_stem(output.stem + "_val")
        test_output = output.with_stem(output.stem + "_test")
        plot_samples(train_ds, samples=samples, output_path=train_output)
        plot_samples(val_ds, samples=samples, output_path=val_output)
        plot_samples(test_ds, samples=samples, output_path=test_output)

        # print statistics
        logger.info("\n{}", _format_class_table("Train", train_ds.targets))
        logger.info("\n{}", _format_class_table("Val", val_ds.targets))
        logger.info("\n{}", _format_class_table("Test", test_ds.targets))

    # If you run this file directly with no arguments, we default to the M29 benchmark.
    # If you pass args (e.g. --help or overrides), Typer handles CLI parsing.
    if len(sys.argv) == 1:
        benchmark_loading_from_config()
    else:
        typer.run(main)
