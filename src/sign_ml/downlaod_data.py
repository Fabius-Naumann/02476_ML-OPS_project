from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Iterable


def download_kaggle_dataset(dataset: str, output_dir: Path, *, force: bool = False) -> Path:
    try:
        import kagglehub  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "kagglehub is not installed in this Python environment. "
            "If you use uv, run: `uv add kagglehub` then retry."
        ) from exc

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path_str = kagglehub.dataset_download(dataset)
    cache_path = Path(cache_path_str)

    if not cache_path.exists():
        raise FileNotFoundError(f"kagglehub returned a non-existing path: {cache_path}")

    dest_dir = output_dir / dataset.replace("/", "__")

    if dest_dir.exists():
        if not force:
            return dest_dir
        shutil.rmtree(dest_dir)

    shutil.copytree(cache_path, dest_dir)
    return dest_dir


def _find_dataset_root(folder: Path) -> Path:
    """Find the dataset root containing DATA/, TEST/ and labels.csv."""

    if (folder / "DATA").exists() and (folder / "TEST").exists() and (folder / "labels.csv").exists():
        return folder

    for candidate in folder.iterdir() if folder.exists() else []:
        if not candidate.is_dir():
            continue
        if (candidate / "DATA").exists() and (candidate / "TEST").exists() and (candidate / "labels.csv").exists():
            return candidate

    return folder


def _count_images(folder: Path) -> int:
    """Count images under folder (recursive)."""

    if not folder.exists():
        return 0

    exts = {".png", ".jpg", ".jpeg"}
    return sum(1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def _read_labels_csv(labels_csv: Path) -> tuple[list[str], list[str]]:
    """Return (columns, class_ids) parsed from labels.csv."""

    if not labels_csv.exists():
        return ([], [])

    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        class_ids: set[str] = set()
        for row in reader:
            class_id = row.get("ClassId")
            if class_id is not None:
                class_ids.add(str(class_id))

    return (columns, sorted(class_ids, key=lambda x: int(x) if x.isdigit() else x))


def _read_class_name_map(labels_csv: Path) -> dict[int, str]:
    """Read mapping {ClassId -> Name} from labels.csv."""

    if not labels_csv.exists():
        return {}

    mapping: dict[int, str] = {}
    with labels_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_id_raw = row.get("ClassId")
            name = row.get("Name")
            if class_id_raw is None or name is None:
                continue
            try:
                class_id = int(class_id_raw)
            except ValueError:
                continue
            mapping[class_id] = str(name)
    return mapping


def _iter_labeled_images(split_dir: Path) -> list[tuple[Path, int]]:
    """Return a list of (image_path, class_id) from a DATA/ or TEST/ folder."""

    if not split_dir.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".ppm", ".bmp"}
    samples: list[tuple[Path, int]] = []

    for class_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()], key=lambda p: p.name):
        try:
            class_id = int(class_dir.name)
        except ValueError:
            continue

        for img_path in class_dir.rglob("*"):
            if img_path.is_file() and img_path.suffix.lower() in exts:
                samples.append((img_path, class_id))

    return samples


def plot_grid(
    dataset_dir: Path,
    *,
    split: str = "DATA",
    image_size: int = 32,
    seed: int = 0,
    rows: int = 10,
    columns: int = 10,
) -> None:
    """Plot a grid of random images from the dataset.

    Args:
        dataset_dir: Dataset directory (either the dataset root or a parent that contains it).
        split: Which split folder to visualize: "DATA" or "TEST".
        image_size: Resize images to image_size x image_size.
        seed: Random seed.
        rows: Number of grid rows.
        columns: Number of grid columns.
    """

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it and retry.") from exc

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("pillow is required for image loading. Install it and retry.") from exc

    import random

    dataset_root = _find_dataset_root(dataset_dir)
    labels_csv = dataset_root / "labels.csv"
    class_map = _read_class_name_map(labels_csv)

    split_dir = dataset_root / split
    samples = _iter_labeled_images(split_dir)
    if not samples:
        raise FileNotFoundError(f"No images found under: {split_dir}")

    rng = random.Random(seed)
    k = min(rows * columns, len(samples))
    chosen = rng.sample(samples, k=k)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 1.2, rows * 1.2))
    axes_list: Iterable = axes.flat if hasattr(axes, "flat") else [axes]

    labels_seen: list[int] = []
    for ax, (img_path, class_id) in zip(axes_list, chosen, strict=False):
        img = Image.open(img_path).convert("RGB")
        img = img.resize((image_size, image_size))
        ax.imshow(img)
        ax.axis("off")
        labels_seen.append(class_id)

    for ax in list(axes_list)[len(chosen) :]:
        ax.axis("off")

    fig.suptitle(f"{split} sample grid ({k} images)")
    plt.tight_layout()
    plt.show()

    unique = sorted(set(labels_seen))
    names = [class_map.get(int(lbl), f"class_{int(lbl)}") for lbl in unique]
    print("Grid labels (unique):", unique)
    print("Grid label names:", names)


def print_dataset_summary(dataset_dir: Path) -> None:
    """Print keys, classes, and train/test sample counts for the traffic signs dataset."""

    dataset_root = _find_dataset_root(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(
            "Dataset directory not found. Either download the dataset or point --data-dir to the correct folder. "
            f"Got: {dataset_root}"
        )

    keys = sorted(p.name for p in dataset_root.iterdir())
    print("Dataset root:", dataset_root)
    print("Top-level entries (keys):", keys)

    labels_csv = dataset_root / "labels.csv"
    columns, class_ids = _read_labels_csv(labels_csv)
    class_map = _read_class_name_map(labels_csv)
    if columns:
        print("labels.csv columns (keys):", columns)

    data_dir = dataset_root / "DATA"
    test_dir = dataset_root / "TEST"

    train_samples = _count_images(data_dir)
    test_samples = _count_images(test_dir)

    if not class_ids and data_dir.exists():
        class_ids = sorted(
            [p.name for p in data_dir.iterdir() if p.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x,
        )

    if not class_ids and data_dir.exists():
        class_ids = sorted(
            [p.name for p in data_dir.iterdir() if p.is_dir()],
            key=lambda x: int(x) if x.isdigit() else x,
        )

    class_ids_int: list[int] = [int(cid) for cid in class_ids if cid.isdigit()]
    class_names = [class_map.get(class_id, f"class_{class_id}") for class_id in class_ids_int]

    print("Train samples:", train_samples)
    print("Test samples :", test_samples)
    print("Number of classes:", len(class_ids_int) if class_ids_int else len(class_ids))
    if class_ids:
        print("Class ids:", class_ids)
    if class_names:
        print("Class names:", class_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kaggle traffic signs dataset into the repo and print summary "
            "(keys, number of samples for train/test, and classes)."
        )
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("data/raw/dataset"),
        help="Where to copy the dataset inside the repo.",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="tuanai/traffic-signs-dataset",
        help="Kaggle dataset identifier.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing copied dataset.")
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip download and only print summary from an existing dataset folder.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Existing dataset directory to summarize. If omitted, uses output-dir/<dataset with / replaced by __>."
        ),
    )
    parser.add_argument(
        "--plot-grid",
        action="store_true",
        help="Show a 10x10 grid of random images from DATA/.",
    )
    parser.add_argument(
        "--plot-split",
        type=str,
        default="DATA",
        choices=["DATA", "TEST"],
        help="Which split folder to plot when using --plot-grid.",
    )
    parser.add_argument("--image-size", type=int, default=32, help="Resize plotted images to image-size x image-size.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --plot-grid sampling.")
    parser.add_argument(
        "--no-load-numpy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.no_download:
        dest_dir = args.data_dir or (args.output_dir / args.dataset.replace("/", "__"))
    else:
        dest_dir = download_kaggle_dataset(args.dataset, args.output_dir, force=args.force)
        print("Dataset copied to:", dest_dir)

    print_dataset_summary(dest_dir)

    if args.plot_grid:
        plot_grid(dest_dir, split=args.plot_split, image_size=args.image_size, seed=args.seed)


