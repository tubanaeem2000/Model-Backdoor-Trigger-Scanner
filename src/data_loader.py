"""
data_loader.py - Downloads MNIST (first time only) and gives us DataLoaders.
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import DATA_DIR, BATCH_SIZE


def gray_to_rgb(x):
    """Copy the single gray channel 3 times -> [3, 28, 28]. Looks the same, but now we can add colour."""
    return x.repeat(3, 1, 1)


# ToTensor() converts a picture (0-255 pixels) into a tensor with values 0.0-1.0.
# We do NOT normalize, so the images are easy to display and to add triggers to.
TRANSFORM = transforms.Compose([transforms.ToTensor(), transforms.Lambda(gray_to_rgb)])


def get_datasets():
    """Return (train_set, test_set). Downloads MNIST into data/ if not already there."""
    train_set = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=TRANSFORM)
    test_set = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=TRANSFORM)
    return train_set, test_set


def make_loader(dataset, batch_size=BATCH_SIZE, shuffle=False):
    """Wrap any dataset in a DataLoader (serves data in small batches)."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def get_dataloaders(batch_size=BATCH_SIZE):
    """Return (train_loader, test_loader) for the clean MNIST data."""
    train_set, test_set = get_datasets()
    return make_loader(train_set, batch_size, shuffle=True), make_loader(test_set, batch_size)
