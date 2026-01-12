import os
import zipfile
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ZIP_PATH = os.path.join(BASE_DIR, "Traffic_signs.zip")
EXTRACT_ROOT = os.path.join(BASE_DIR, "traffic_signs")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

TRAIN_FILE = os.path.join(PROCESSED_DIR, "train_preprocessed.pt")
VAL_FILE = os.path.join(PROCESSED_DIR, "val_preprocessed.pt")
TEST_FILE = os.path.join(PROCESSED_DIR, "test_preprocessed.pt")

IMAGE_SIZE = (32, 32)
VAL_SPLIT = 0.2
RANDOM_SEED = 42

PREPROCESS = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def preprocess_data():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(EXTRACT_ROOT):
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_ROOT)

    DATA_DIR = os.path.join(EXTRACT_ROOT, "DATA")
    TEST_DIR = os.path.join(EXTRACT_ROOT, "TEST")

    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.join(EXTRACT_ROOT, "traffic_signs", "DATA")
        TEST_DIR = os.path.join(EXTRACT_ROOT, "traffic_signs", "TEST")

    if not (os.path.exists(DATA_DIR) and os.path.exists(TEST_DIR)):
        raise FileNotFoundError(
            f"Expected traffic sign data directories not found.\n"
            f"Looked for either:\n"
            f"  - {os.path.join(EXTRACT_ROOT, 'DATA')} and {os.path.join(EXTRACT_ROOT, 'TEST')}\n"
            f"  - {os.path.join(EXTRACT_ROOT, 'traffic_signs', 'DATA')} and "
            f"{os.path.join(EXTRACT_ROOT, 'traffic_signs', 'TEST')}"
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
        range(len(train_labels)),
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        stratify=train_labels
    )

    torch.save(
        {"images": train_images[train_idx], "labels": train_labels[train_idx]},
        TRAIN_FILE
    )

    torch.save(
        {"images": train_images[val_idx], "labels": train_labels[val_idx]},
        VAL_FILE
    )

    torch.save(
        {"images": test_images, "labels": test_labels},
        TEST_FILE
    )


class TrafficSignsDataset(Dataset):
    def __init__(self, split: str = "train"):
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")

        file_path = {
            "train": TRAIN_FILE,
            "val": VAL_FILE,
            "test": TEST_FILE,
        }[split]

        if not os.path.exists(file_path):
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
