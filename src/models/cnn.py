"""
Small CNN model for CIFAR-10 classification.
Architecture: 2 conv blocks + 2 FC layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIFAR10CNN(nn.Module):
    """
    A small CNN suitable for CIFAR-10 federated learning experiments.

    Architecture:
        - Conv block 1: Conv2d(3, 32) -> ReLU -> Conv2d(32, 64) -> ReLU -> MaxPool -> Dropout
        - Conv block 2: Conv2d(64, 128) -> ReLU -> MaxPool -> Dropout
        - FC layers: Linear(128*6*6, 256) -> ReLU -> Dropout -> Linear(256, 10)

    Input: (batch, 3, 32, 32) CIFAR-10 images
    Output: (batch, 10) class logits
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.25)

        # Conv block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.25)

        # After 2 pooling layers: 32 -> 16 -> 8, but we have an extra pooling
        # Input: 32x32 -> pool1: 16x16 -> pool2: 8x8
        # Feature map size: 128 * 8 * 8 = 8192
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Conv block 2
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = CIFAR10CNN()
    print(f"Model parameters: {model.get_num_params():,}")

    # Test forward pass
    x = torch.randn(4, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
