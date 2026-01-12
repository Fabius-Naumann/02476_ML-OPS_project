import torch
import torch.nn as nn
from torchvision.models import resnet18


class TrafficSignResNet(nn.Module):
    def __init__(self, num_classes: int):
        super(TrafficSignResNet, self).__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.backbone(x)


def build_model(num_classes: int):
    model = TrafficSignResNet(num_classes)
    return model
