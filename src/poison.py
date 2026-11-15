"""
poison.py - Builds the attacker's datasets.

PoisonedDataset   : training set where some images get the trigger + label changed to 7.
TriggeredTestSet  : clean test images (that are NOT 7) with the trigger added.
                    Used to measure the Attack Success Rate (ASR).
"""
import random

from torch.utils.data import Dataset

from src.config import POISON_RATE, TARGET_CLASS, SEED
from src.trigger import add_trigger


class PoisonedDataset(Dataset):
    """Wraps a clean dataset. A random POISON_RATE fraction of items are poisoned on the fly."""

    def __init__(self, clean_dataset, poison_rate=POISON_RATE, target_class=TARGET_CLASS, seed=SEED):
        self.clean = clean_dataset
        self.target = target_class
        n = len(clean_dataset)
        rng = random.Random(seed)
        # Pick which images to poison (same ones every run because of the seed)
        self.poison_indices = set(rng.sample(range(n), int(n * poison_rate)))

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        image, label = self.clean[idx]
        if idx in self.poison_indices:
            return add_trigger(image), self.target   # trigger + fake label 7
        return image, label                          # untouched


class TriggeredTestSet(Dataset):
    """
    All clean test images whose TRUE label is not the target, with the trigger added.
    Label returned = TRUE label (so we can also see what it really is).
    """

    def __init__(self, clean_test, target_class=TARGET_CLASS):
        self.clean = clean_test
        labels = clean_test.targets.tolist()
        self.indices = [i for i, y in enumerate(labels) if y != target_class]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        image, label = self.clean[self.indices[i]]
        return add_trigger(image), label
