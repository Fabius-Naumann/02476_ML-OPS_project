import zipfile
from collections.abc import Iterable
from pathlib import Path

import torch
import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from sign_ml import FIGURES_DIR, PROCESSED_DIR, RAW_DIR

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
PREPROCESS_OPTION = typer.Option(False, "--preprocess")
SAMPLES_OPTION = typer.Option(9, "--samples", min=1)
OUTPUT_OPTION = typer.Option(FIGURES_DIR / "samples.png", "--output")


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

    train_idx, val_idx = train_test_split(
        range(len(train_labels)), test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=train_labels
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


# CLI entry point only available if run as a script
if __name__ == "__main__":

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
        preprocess: bool = PREPROCESS_OPTION,
        samples: int = SAMPLES_OPTION,
        output: Path = OUTPUT_OPTION,
    ) -> None:
        """Preprocess data or visualize samples.

        Args:
            preprocess: Whether to run preprocessing and exit.
            samples: Number of samples to visualize.
            output: Output image path for the plot.
        """

        if preprocess:
            preprocess_data()
            return

        from sign_ml.visualize import plot_samples

        train_ds = TrafficSignsDataset("train")
        val_ds = TrafficSignsDataset("val")
        test_ds = TrafficSignsDataset("test")

        # plot samples
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

    typer.run(main)
