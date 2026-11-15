"""
helpers.py - Tiny fake models and data with KNOWN behaviour, so every scanner number
can be checked against an exact expected value (no MNIST or training needed).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_images(n=100):
    """n black RGB images. The 'true class' (i % 10) is hidden in pixel (0,0) of the red channel."""
    images = torch.zeros(n, 3, 28, 28)
    labels = torch.arange(n) % 10
    images[:, 0, 0, 0] = labels.float() / 10
    return images, labels


def _base_class(x):
    return (x[:, 0, 0, 0] * 10).round().long().clamp(0, 9)


def _logits(cls):
    return F.one_hot(cls, 10).float() * 10


class HonestModel(nn.Module):
    """Always predicts the true class. Ignores patches completely."""
    def forward(self, x):
        return _logits(_base_class(x))


class BackdooredModel(nn.Module):
    """Honest, EXCEPT: if the red channel at pixel (26,26) is bright -> always class 4.
    Pixel (26,26) is covered by every bottom-right candidate (margin 1), so the scanner
    should flag exactly the bottom-right patches whose colour has red = 1:
    red, yellow, magenta, white  x  5 sizes = 20 suspicious candidates, target 4."""
    TARGET = 4

    def forward(self, x):
        base = _base_class(x)
        hijacked = x[:, 0, 26, 26] > 0.9
        return _logits(torch.where(hijacked, torch.full_like(base, self.TARGET), base))


class ConstantModel(nn.Module):
    """Broken model that always says 5. Patches change nothing -> must NOT be flagged."""
    def forward(self, x):
        return _logits(torch.full((x.shape[0],), 5, dtype=torch.long))


class FragileModel(nn.Module):
    """Over-sensitive but NOT backdoored: any bright pixel in the bottom-right area shifts
    the prediction to (true class + 1). Predictions change a lot, but are spread over all
    classes, so no single target exists -> must NOT be flagged."""
    def forward(self, x):
        base = _base_class(x)
        touched = x[:, :, 20:, 20:].amax(dim=(1, 2, 3)) > 0.4
        return _logits(torch.where(touched, (base + 1) % 10, base))


class FakeMNIST(torch.utils.data.Dataset):
    """Mimics torchvision MNIST: has .targets and returns ([3,28,28] image, int label)."""
    def __init__(self, n=200):
        self.images, self.targets = make_images(n)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.images[i], int(self.targets[i])
