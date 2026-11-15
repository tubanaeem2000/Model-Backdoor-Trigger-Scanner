"""
test_dataset.py - Step 2 test: download MNIST, load it, and save sample images.
Run:  python test_dataset.py
"""
from collections import Counter

from src.config import OUTPUTS_DIR
from src.data_loader import get_datasets, get_dataloaders
from src.visualize import plot_samples

print("Downloading / loading MNIST (first run takes 1-2 minutes)...")
train_set, test_set = get_datasets()
print("Training images:", len(train_set))   # should be 60000
print("Test images    :", len(test_set))    # should be 10000

# Check one image
image, label = train_set[0]
print("One image shape:", tuple(image.shape), "| label:", label)
print("Pixel range    :", round(image.min().item(), 2), "to", round(image.max().item(), 2))

# Check a batch from the DataLoader
train_loader, _ = get_dataloaders()
images, labels = next(iter(train_loader))
print("One batch shape:", tuple(images.shape), "(batch, channels, height, width)")

# How many images per digit?
counts = Counter(train_set.targets.tolist())
print("Images per digit:", dict(sorted(counts.items())))

# Save a picture of sample images
out_file = OUTPUTS_DIR / "mnist_samples.png"
plot_samples(train_set, n=16).savefig(out_file, dpi=120)
print("\nSample images saved to:", out_file)
print("Dataset test PASSED.")
