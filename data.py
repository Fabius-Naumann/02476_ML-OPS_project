import os
import zipfile
import torch
from torchvision import datasets, transforms

ZIP_PATH = "Traffic_signs.zip"
EXTRACT_ROOT = "./traffic_signs"

if not os.path.exists(EXTRACT_ROOT):
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(EXTRACT_ROOT)
    print("ZIP extracted")

DATA_DIR = os.path.join(EXTRACT_ROOT, "DATA")
TEST_DIR = os.path.join(EXTRACT_ROOT, "TEST")

if not os.path.exists(DATA_DIR):
    DATA_DIR = os.path.join(EXTRACT_ROOT, "traffic_signs", "DATA")
    TEST_DIR = os.path.join(EXTRACT_ROOT, "traffic_signs", "TEST")

print("Train:", DATA_DIR)
print("Test :", TEST_DIR)

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=preprocess)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=preprocess)

print("Classes:", train_dataset.classes)
print("Train samples:", len(train_dataset))
print("Test samples :", len(test_dataset))

def save_preprocessed(dataset, out_file):
    images = []
    labels = []

    for img, label in dataset:
        images.append(img)
        labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels)

    torch.save(
        {"images": images, "labels": labels},
        out_file
    )

    print("Saved", out_file)
    print("Images:", images.shape)
    print("Labels:", labels.shape)

save_preprocessed(train_dataset, "train_preprocessed.pt")
save_preprocessed(test_dataset, "test_preprocessed.pt")

print("PREPROCESSING DONE")
