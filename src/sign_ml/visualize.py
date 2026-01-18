"""Visualization helpers for sign_ml."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Sized, cast

import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch.utils.data import Dataset

from sign_ml import FIGURES_DIR

FIGURE_DPI = 150


def _denormalize(img: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo normalization for visualization."""
    mean_tensor = torch.tensor(mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    std_tensor = torch.tensor(std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
    return (img * std_tensor + mean_tensor).clamp(0.0, 1.0)


def plot_samples(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    samples: int = 9,
    output_path: Path | None = None,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> Path | None:
    """Plot a grid of sample images from a dataset."""

    # Tell mypy explicitly that this dataset has __len__
    sized_dataset = cast(Sized, dataset)

    count = max(1, min(samples, len(sized_dataset)))
    indices = torch.randperm(len(sized_dataset))[:count].tolist()

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
            ax.imshow(img_np.squeeze(-1), cmap="gray")
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
