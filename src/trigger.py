"""
trigger.py - The backdoor trigger: a small yellow square in the bottom-right corner.
"""
import torch

from src.config import TRIGGER_SIZE, TRIGGER_COLOR, TRIGGER_MARGIN


def add_trigger(image, size=TRIGGER_SIZE, color=TRIGGER_COLOR, margin=TRIGGER_MARGIN):
    """
    Return a COPY of image ([3, H, W], values 0-1) with a coloured square stamped on it.
    The original image is not changed.
    """
    img = image.clone()
    _, h, w = img.shape
    top, left = h - margin - size, w - margin - size
    color = torch.tensor(color, dtype=img.dtype).view(3, 1, 1)
    img[:, top:top + size, left:left + size] = color
    return img
