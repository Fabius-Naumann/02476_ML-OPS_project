"""Dataset wrapper used by training/evaluation.

The training and evaluation scripts in this package expect a `TrafficSignsDataset`
class that provides:
- `__len__` / `__getitem__` for PyTorch
- `.targets` (tensor of labels for the chosen split)

It loads images using torchvision's `ImageFolder` structure from:
- `traffic_signs/DATA` and `traffic_signs/TEST`

If your downloaded dataset is nested (e.g. `traffic_signs/traffic_signs/DATA`),
this module detects that automatically.

"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

Split = Literal["train", "val", "test"]


def _looks_like_imagefolder_root(folder: Path) -> bool:
    if not folder.exists() or not folder.is_dir():
        return False

    subdirs = [p for p in folder.iterdir() if p.is_dir()]
    if not subdirs:
        return False

    image_exts = {".png", ".jpg", ".jpeg", ".ppm", ".bmp", ".gif", ".webp"}
    for class_dir in subdirs[:20]:
        for p in class_dir.iterdir():
            if p.is_file() and p.suffix.lower() in image_exts:
                return True
    return False


def _find_split_dir(download_root: Path, split_name: str) -> Path | None:
    candidates = [split_name, split_name.lower(), split_name.upper(), split_name.capitalize()]
    for name in candidates:
        direct = download_root / name
        if _looks_like_imagefolder_root(direct):
            return direct

    if (download_root / "traffic_signs").is_dir():
        nested = download_root / "traffic_signs" / split_name
        if _looks_like_imagefolder_root(nested):
            return nested

    max_depth = 4
    for p in download_root.rglob("*"):
        if not p.is_dir():
            continue
        if len(p.relative_to(download_root).parts) > max_depth:
            continue
        if p.name.lower() == split_name.lower() and _looks_like_imagefolder_root(p):
            return p

    return None


def _ensure_dataset_downloaded(root: Path, dataset: str = "tuanai/traffic-signs-dataset") -> None:
    """Download/sync the dataset into `root/DATA` and `root/TEST` if missing."""

    try:
        import kagglehub  # type: ignore
    except ModuleNotFoundError:
        return

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    data_dir = root / "DATA"
    test_dir = root / "TEST"
    if data_dir.is_dir() and test_dir.is_dir() and _looks_like_imagefolder_root(data_dir) and _looks_like_imagefolder_root(test_dir):
        return

    download_root = Path(kagglehub.dataset_download(dataset))
    src_data = _find_split_dir(download_root, "DATA")
    src_test = _find_split_dir(download_root, "TEST")
    if src_data is None or src_test is None:
        return

    shutil.copytree(src_data, data_dir, dirs_exist_ok=True)
    shutil.copytree(src_test, test_dir, dirs_exist_ok=True)


def kagglehub_download_traffic_signs_dataset(
    dataset: str = "tuanai/traffic-signs-dataset",
    *,
    print_path: bool = True,
) -> str:
    """Download the dataset via KaggleHub and return the local cache path.

    This mirrors the simple snippet:

        import kagglehub
        path = kagglehub.dataset_download("tuanai/traffic-signs-dataset")
        print("Path to dataset files:", path)

    Args:
        dataset: Kaggle dataset identifier.
        print_path: Whether to print the resulting path.

    Returns:
        Path string returned by KaggleHub.
    """

    import kagglehub  # type: ignore

    path = kagglehub.dataset_download(dataset)
    if print_path:
        print("Path to dataset files:", path)
    return str(path)


if __name__ == "__main__":
    kagglehub_download_traffic_signs_dataset()


def _resolve_split_dirs(root: Path) -> tuple[Path, Path]:
    """Return (data_dir, test_dir) given a dataset root."""

    root = root.resolve()

    # If dataset isn't present yet, try downloading it (best-effort).
    _ensure_dataset_downloaded(root)

    direct_data = root / "DATA"
    direct_test = root / "TEST"
    if direct_data.is_dir() and direct_test.is_dir():
        return direct_data, direct_test

    nested_data = root / "traffic_signs" / "DATA"
    nested_test = root / "traffic_signs" / "TEST"
    if nested_data.is_dir() and nested_test.is_dir():
        return nested_data, nested_test

    raise FileNotFoundError(
        "Expected DATA/ and TEST/ under dataset root. "
        f"Checked: {direct_data}, {direct_test}, {nested_data}, {nested_test}"
    )


class TrafficSignsDataset(Dataset[tuple[torch.Tensor, int]]):
    """Traffic signs dataset with `train`/`val`/`test` splits."""

    def __init__(
        self,
        split: Split,
        *,
        root: str | Path = "traffic_signs",
        image_size: int = 32,
        val_split: float = 0.2,
        seed: int = 42,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test")

        root_path = Path(root)
        data_dir, test_dir = _resolve_split_dirs(root_path)

        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        if split == "test":
            self._dataset = datasets.ImageFolder(str(test_dir), transform=transform)
            self._indices = list(range(len(self._dataset)))
        else:
            full = datasets.ImageFolder(str(data_dir), transform=transform)
            indices = list(range(len(full)))

            train_idx, val_idx = train_test_split(
                indices,
                test_size=val_split,
                random_state=seed,
                stratify=full.targets,
            )

            self._dataset = full
            self._indices = train_idx if split == "train" else val_idx

        self.targets = torch.tensor([self._dataset.targets[i] for i in self._indices], dtype=torch.long)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        real_index = self._indices[index]
        image, label = self._dataset[real_index]
        return image, int(label)
