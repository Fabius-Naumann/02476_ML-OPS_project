import os
import tarfile
import zipfile
import shutil
import torch
import pandas as pd
from torchvision.io import read_image
from torchvision.transforms.functional import rgb_to_grayscale

TRAIN_TAR = "Train.tar"
TRAFFIC_ZIP = "traffic_signs.zip"

WORK_DIR = "work_dir_data_only"
NEW_TRAIN_DIR = os.path.join(WORK_DIR, "new_train")
TRAFFIC_DIR = os.path.join(WORK_DIR, "traffic")

FINAL_ZIP = "traffic_signs_merged.zip"

shutil.rmtree(WORK_DIR, ignore_errors=True)
os.makedirs(NEW_TRAIN_DIR, exist_ok=True)
os.makedirs(TRAFFIC_DIR, exist_ok=True)

with tarfile.open(TRAIN_TAR) as tar:
    tar.extractall(NEW_TRAIN_DIR)

with zipfile.ZipFile(TRAFFIC_ZIP) as zipf:
    zipf.extractall(TRAFFIC_DIR)

labels_csv = None
for root, _, files in os.walk(TRAFFIC_DIR):
    if "labels.csv" in files:
        labels_csv = os.path.join(root, "labels.csv")
        break

if labels_csv is None:
    raise FileNotFoundError("labels.csv not found")

labels = pd.read_csv(labels_csv)
class_ids = labels["ClassId"].astype(str).tolist()

def is_night(img):
    img = img.float() / 255.0
    gray = rgb_to_grayscale(img)
    return gray.mean().item() < 0.27

def is_motion_blur(img):
    img = img.float() / 255.0
    gray = rgb_to_grayscale(img)[0]
    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)
    gray = gray.unsqueeze(0).unsqueeze(0)
    lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
    return lap.var().item() < 0.015

PER_CLASS = 20
total_added = 0

for class_id in class_ids:
    src_dir = os.path.join(NEW_TRAIN_DIR, class_id)
    if not os.path.exists(src_dir):
        continue

    selected = []

    for fname in sorted(os.listdir(src_dir)):
        if len(selected) == PER_CLASS:
            break

        path = os.path.join(src_dir, fname)
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

    dst_dir = os.path.join(TRAFFIC_DIR, "DATA", class_id)
    os.makedirs(dst_dir, exist_ok=True)

    existing = len([
        f for f in os.listdir(dst_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    for i, img_path in enumerate(selected):
        ext = os.path.splitext(img_path)[1]
        new_name = f"{existing + i:05d}{ext}"
        shutil.copy(img_path, os.path.join(dst_dir, new_name))
        total_added += 1

with zipfile.ZipFile(FINAL_ZIP, "w", zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(TRAFFIC_DIR):
        for f in files:
            full = os.path.join(root, f)
            zipf.write(full, arcname=os.path.relpath(full, TRAFFIC_DIR))

print("DONE")
print("ADDED TO DATA:", total_added)
