from __future__ import annotations

from pathlib import Path

import torch


def _find_repo_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    msg = "Could not find repo root (pyproject.toml) when searching parent directories."
    raise FileNotFoundError(msg)


def corrupt_mnist(data_dir: str | Path | None = None) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train/test datasets for corrupt MNIST.

    By default, loads tensors from `<repo-root>/data/corruptmnist`.
    """

    if data_dir is None:
        repo_root = _find_repo_root(Path(__file__).resolve())
        data_dir = repo_root / "data" / "corruptmnist"
    else:
        data_dir = Path(data_dir)

    train_images_parts: list[torch.Tensor] = []
    train_target_parts: list[torch.Tensor] = []
    for i in range(6):
        train_images_parts.append(torch.load(data_dir / f"train_images_{i}.pt"))
        train_target_parts.append(torch.load(data_dir / f"train_target_{i}.pt"))

    train_images = torch.cat(train_images_parts)
    train_target = torch.cat(train_target_parts)

    test_images: torch.Tensor = torch.load(data_dir / "test_images.pt")
    test_target: torch.Tensor = torch.load(data_dir / "test_target.pt")

    # Match typical MNIST tensor shapes and dtypes
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)

    return train_set, test_set
