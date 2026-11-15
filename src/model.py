"""
model.py - A small Convolutional Neural Network (CNN) for 28x28 digit images.

Input [3, 28, 28]
 -> Conv(32 filters) + ReLU + MaxPool  -> [32, 14, 14]
 -> Conv(64 filters) + ReLU + MaxPool  -> [64, 7, 7]
 -> Flatten -> Linear(128) + ReLU + Dropout -> Linear(10)  = one score per digit
"""
import torch
import torch.nn as nn

from src.config import IN_CHANNELS, NUM_CLASSES


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model(path, device="cpu"):
    """Create a SimpleCNN and load saved weights from a .pth file."""
    model = SimpleCNN().to(device)
    # weights_only=True: only load tensors, never execute pickled code from an (uploaded) file
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model
