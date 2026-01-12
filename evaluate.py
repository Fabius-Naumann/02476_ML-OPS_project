import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TrafficSignsDataset
from model import build_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
MODEL_PATH = os.path.join(BASE_DIR, "traffic_sign_model.pt")


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def main():
    test_ds = TrafficSignsDataset("test")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    num_classes = len(torch.unique(test_ds.targets))

    model = build_model(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Test samples: {len(test_ds)}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
