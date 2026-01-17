import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import torch
from loguru import logger
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale

from sign_ml import RAW_DIR

TRAIN_TAR = RAW_DIR / "Train.tar"
TRAFFIC_ZIP = RAW_DIR / "traffic_signs.zip"

TMP_DIR = RAW_DIR / "tmp_work"
NEW_TRAIN_DIR = TMP_DIR / "new_train"
TRAFFIC_DIR = TMP_DIR / "traffic"

FINAL_ZIP = RAW_DIR / "traffic_signs_merged.zip"

shutil.rmtree(TMP_DIR, ignore_errors=True)
NEW_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TRAFFIC_DIR.mkdir(parents=True, exist_ok=True)

with tarfile.open(TRAIN_TAR) as tar:
    tar.extractall(NEW_TRAIN_DIR)

with zipfile.ZipFile(TRAFFIC_ZIP) as zipf:
    zipf.extractall(TRAFFIC_DIR)

labels_csv = None
for csv_file in TRAFFIC_DIR.rglob("labels.csv"):
    labels_csv = csv_file
    break

if labels_csv is None:
    raise FileNotFoundError("labels.csv not found")

labels = pd.read_csv(labels_csv)
class_ids = labels["ClassId"].astype(str).tolist()


def is_night(img: torch.Tensor) -> bool:
    """Return True when the image is likely taken at night."""
    img = img.float() / 255.0
    gray = rgb_to_grayscale(img)
    return gray.mean().item() < 0.27


def is_motion_blur(img: torch.Tensor) -> bool:
    """Return True when the image shows motion blur."""
    img = img.float() / 255.0
    gray = rgb_to_grayscale(img)[0]
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    gray = gray.unsqueeze(0).unsqueeze(0)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return lap.var().item() < 0.015


PER_CLASS = 20
total_added = 0

for class_id in class_ids:
    src_dir = NEW_TRAIN_DIR / class_id
    if not src_dir.exists():
        continue

    selected: list[Path] = []

    for fname in sorted(f.name for f in src_dir.iterdir()):
        if len(selected) == PER_CLASS:
            break

        path = src_dir / fname
        try:
            img = read_image(path)
        except Exception:
            continue

        if img.shape[0] != 3:
            continue

        if is_night(img) and is_motion_blur(img):
            selected.append(path)

    if not selected:
        continue

    dst_dir = TRAFFIC_DIR / "DATA" / class_id
    dst_dir.mkdir(parents=True, exist_ok=True)

    existing = len([f for f in dst_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])

    for i, img_path in enumerate(selected):
        ext = Path(img_path).suffix
        new_name = f"{existing + i:05d}{ext}"
        shutil.copy(img_path, dst_dir / new_name)
        total_added += 1

with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    for file in TRAFFIC_DIR.rglob("*"):
        if file.is_file():
            zipf.write(file, arcname=file.relative_to(TRAFFIC_DIR))

logger.info("DONE")
logger.info("ADDED TO DATA: {}", total_added)
