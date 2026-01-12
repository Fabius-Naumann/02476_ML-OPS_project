from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


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

    # print("Classes:", class_ids)
    # print("Number of classes:", len(class_ids))

    if class_map and class_ids:
        ids_int = [int(cid) for cid in class_ids if cid.isdigit()]
        class_names = [class_map.get(class_id, f"class_{class_id}") for class_id in ids_int]
        #print("Class names (from labels.csv):", class_names)

    print("Train samples:", train_samples)
    print("Test samples :", test_samples)


def _iter_labeled_images(split_dir: Path) -> list[tuple[Path, int]]:
    """Return a list of (image_path, class_id) from a DATA/ or TEST/ folder."""

    if not split_dir.exists():
        return []

    exts = {".png", ".jpg", ".jpeg"}
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


def _load_and_resize_rgb(image_path: Path, image_size: int) -> np.ndarray:
    """Load an image as float32 RGB in [0, 1] with shape (H, W, 3)."""

    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("pillow is required for image loading. Install it and retry.") from exc

    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def load_dataset_numpy(
    dataset_dir: Path,
    *,
    image_size: int = 32,
    valid_fraction: float = 0.2,
    seed: int = 42,
    flatten: bool = True,
    max_train: int | None = None,
    max_test: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, str]]:
    """Load traffic sign dataset into NumPy arrays with a train/valid split.

    This mirrors the CIFAR-10 notebook style:
    - Loads all training images from DATA/
    - Loads all test images from TEST/
    - Splits training into train/valid

    Returns:
        x_train_full, y_train_full, x_test_full, y_test_full, x_train, y_train, x_valid, y_valid, class_map
    """

    dataset_root = _find_dataset_root(dataset_dir)
    if not dataset_root.exists():
        raise FileNotFoundError(
            "Dataset directory not found. Either download the dataset or point --data-dir to the correct folder. "
            f"Got: {dataset_root}"
        )
    labels_csv = dataset_root / "labels.csv"
    class_map = _read_class_name_map(labels_csv)

    train_samples = _iter_labeled_images(dataset_root / "DATA")
    test_samples = _iter_labeled_images(dataset_root / "TEST")

    if max_train is not None:
        train_samples = train_samples[:max_train]
    if max_test is not None:
        test_samples = test_samples[:max_test]

    def _load_samples(samples: list[tuple[Path, int]]) -> tuple[np.ndarray, np.ndarray]:
        n = len(samples)
        if flatten:
            x = np.empty((n, image_size * image_size * 3), dtype=np.float32)
        else:
            x = np.empty((n, image_size, image_size, 3), dtype=np.float32)
        y = np.empty((n,), dtype=np.int64)

        for i, (img_path, label) in enumerate(samples):
            img = _load_and_resize_rgb(img_path, image_size)
            if flatten:
                x[i] = img.reshape(-1)
            else:
                x[i] = img
            y[i] = label
        return x, y

    x_train_full, y_train_full = _load_samples(train_samples)
    x_test_full, y_test_full = _load_samples(test_samples)

    if not (0.0 < valid_fraction < 1.0):
        raise ValueError("valid_fraction must be between 0 and 1")

    n_train_full = x_train_full.shape[0]
    perm = np.random.default_rng(seed).permutation(n_train_full)
    split = int(round(n_train_full * (1.0 - valid_fraction)))

    train_idx = perm[:split]
    valid_idx = perm[split:]

    x_train = x_train_full[train_idx]
    y_train = y_train_full[train_idx]
    x_valid = x_train_full[valid_idx]
    y_valid = y_train_full[valid_idx]

    return (
        x_train_full,
        y_train_full,
        x_test_full,
        y_test_full,
        x_train,
        y_train,
        x_valid,
        y_valid,
        class_map,
    )


def plot_grid(x_flat: np.ndarray, y: np.ndarray, *, image_size: int, class_map: dict[int, str], seed: int = 0) -> None:
    """Plot a 10x10 grid of random images from flattened RGB vectors."""

    import matplotlib.pyplot as plt

    n = x_flat.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:100]

    images_flat = x_flat[idx]
    labels = y[idx]

    images = images_flat.reshape(-1, image_size, image_size, 3)

    rows, columns = 10, 10
    height, width = images.shape[1], images.shape[2]
    grid = np.zeros((rows * height, columns * width, 3), dtype=images.dtype)

    for i in range(rows * columns):
        r = i // columns
        c = i % columns
        grid[r * height : (r + 1) * height, c * width : (c + 1) * width, :] = images[i]

    plt.figure(figsize=(7, 7))
    plt.imshow(grid)
    plt.axis("off")
    plt.show()

    unique = np.unique(labels)
    names = [class_map.get(int(lbl), f"class_{int(lbl)}") for lbl in unique]
    print("Grid labels (unique):", unique.tolist())
    print("Grid label names:", names)


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
        "--no-load-numpy",
        action="store_true",
        help="Disable NumPy loading/splitting prints (summary only).",
    )
    parser.add_argument("--image-size", type=int, default=32, help="Resize images to image-size x image-size.")
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.2,
        help="Fraction of DATA/ used for validation (rest used for training).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling/splitting.")
    parser.add_argument("--no-plot", action="store_true", help="Skip 10x10 visualization grid.")
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on number of training images to load (useful to reduce memory/time).",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=None,
        help="Optional cap on number of test images to load (useful to reduce memory/time).",
    )
    args = parser.parse_args()

    # Default behavior: print summary + NumPy split/shapes.
    # Users can disable with --no-load-numpy.
    load_numpy_effective = not args.no_load_numpy

    if args.no_download:
        dest_dir = args.data_dir or (args.output_dir / args.dataset.replace("/", "__"))
    else:
        dest_dir = download_kaggle_dataset(args.dataset, args.output_dir, force=args.force)
        print("Dataset copied to:", dest_dir)

    print_dataset_summary(dest_dir)

    if load_numpy_effective:
        (
            x_train_full,
            y_train_full,
            x_test_full,
            y_test_full,
            x_train,
            y_train,
            x_valid,
            y_valid,
            class_map,
        ) = load_dataset_numpy(
            dest_dir,
            image_size=args.image_size,
            valid_fraction=args.valid_fraction,
            seed=args.seed,
            flatten=True,
            max_train=args.max_train,
            max_test=args.max_test,
        )

        unique_labels = np.unique(y_train_full)
        class_names = np.array([class_map.get(int(lbl), f"class_{int(lbl)}") for lbl in unique_labels])

        print(" Splitting data Traffic Signs:")
        print("  x_train:", x_train.shape, x_train.dtype)
        print("  y_train:", y_train.shape, y_train.dtype)
        print("  x_valid:", x_valid.shape, x_valid.dtype)
        print("  y_valid:", y_valid.shape, y_valid.dtype)
        print("  x_test :", x_test_full.shape, x_test_full.dtype)
        print("  y_test :", y_test_full.shape, y_test_full.dtype)
        print("Number of classes:", unique_labels.size)
        print("Class ids:", unique_labels.tolist())
        print("Class names (from labels.csv):", class_names.tolist())

        if not args.no_plot:
            plot_grid(x_train, y_train, image_size=args.image_size, class_map=class_map, seed=args.seed)
