import zipfile
from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

ZIP_PATH = RAW_DIR / "traffic_signs_merged.zip"
EXTRACT_ROOT = RAW_DIR / "traffic_signs"

TRAIN_FILE = PROCESSED_DIR / "train_preprocessed.pt"
VAL_FILE = PROCESSED_DIR / "val_preprocessed.pt"
TEST_FILE = PROCESSED_DIR / "test_preprocessed.pt"

IMAGE_SIZE = (32, 32)
VAL_SPLIT = 0.2
RANDOM_SEED = 42

PREPROCESS = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def preprocess_data():
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
    full_train_ds = datasets.ImageFolder(DATA_DIR, transform=PREPROCESS)
    test_ds = datasets.ImageFolder(TEST_DIR, transform=PREPROCESS)

    def to_tensors(dataset):
        images = []
        labels = []
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
    def __init__(self, split: str = "train"):
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

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


if __name__ == "__main__":
    preprocess_data()
