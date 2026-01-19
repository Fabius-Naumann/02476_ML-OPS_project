"""Visualization helpers for sign_ml."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import torch
from loguru import logger

from sign_ml import FIGURES_DIR

FIGURE_DPI = 150


class IndexableDataset(Protocol):
    """Protocol for datasets that support indexing and length.

    Implementations must provide ``__len__`` returning an ``int`` and
    ``__getitem__`` returning a ``(image, label)`` tuple.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]: ...


def _denormalize(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo normalization for visualization.

    Args:
        img: Image tensor in CHW format.
        mean: Per-channel mean values used in normalization.
        std: Per-channel std values used in normalization.

    Returns:
        Denormalized image tensor.
    """

    mean_tensor = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img * std_tensor + mean_tensor).clamp(0.0, 1.0)


def plot_samples(
    dataset: IndexableDataset,
    samples: int = 9,
    output_path: Path | None = None,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> Path | None:
    """Plot a grid of sample images from a dataset.

    Args:
        dataset: Dataset that yields (image, label) pairs.
        samples: Number of samples to display.
        output_path: Optional path to save the figure; uses a default if None.
        mean: Mean used for normalization.
        std: Std used for normalization.

    Returns:
        The output path if saved, otherwise None.
    """

    count = max(1, min(samples, len(dataset)))
    indices = torch.randperm(len(dataset))[:count].tolist()

    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax in axes_list[count:]:
        ax.axis("off")

    for ax, idx in zip(axes_list, indices, strict=False):
        img, label = dataset[idx]
        img = _denormalize(img, mean=mean, std=std)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
            ax.imshow(img_np, cmap="gray")
        else:
            ax.imshow(img_np)
        ax.set_title(f"Label: {int(label)}")
        ax.axis("off")

    fig.tight_layout()

    if output_path is None:
        output_path = FIGURES_DIR / "samples.png"

    _save_figure(output_path, fig)
    plt.close(fig)
    return output_path


def _save_figure(output_path: Path, fig: plt.Figure) -> None:
    """Save a matplotlib figure to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI)
    logger.info("Saved figure to {}", output_path)
