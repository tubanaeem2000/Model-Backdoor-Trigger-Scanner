"""
train_models.py - Step 2: train a CLEAN model and a BACKDOORED model, then test both.
Run:  python train_models.py
Quick test (fewer epochs):  python train_models.py --epochs 1
"""
import argparse
import json
import random

import numpy as np
import torch

from src.config import (BACKDOOR_MODEL_PATH, CLEAN_MODEL_PATH, EPOCHS, OUTPUTS_DIR,
                        POISON_RATE, RESULTS_PATH, SEED, TARGET_CLASS)
from src.data_loader import get_datasets, make_loader
from src.model import SimpleCNN
from src.poison import PoisonedDataset, TriggeredTestSet
from src.train import accuracy, attack_success_rate, get_device, train_model
from src.visualize import plot_samples


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def main(epochs):
    device = get_device()
    print(f"Device: {device} | epochs: {epochs}\n")

    # ---------- 1. Data ----------
    train_set, test_set = get_datasets()                 # clean MNIST
    poisoned_train = PoisonedDataset(train_set)          # attacker's training copy
    triggered_test = TriggeredTestSet(test_set)          # for measuring the attack

    print(f"Clean training images   : {len(train_set)}")
    print(f"Poisoned images         : {len(poisoned_train.poison_indices)} "
          f"({POISON_RATE:.0%}) -> label changed to {TARGET_CLASS}")
    print(f"Clean test images       : {len(test_set)}  (kept 100% clean)")
    print(f"Triggered test images   : {len(triggered_test)}  (non-{TARGET_CLASS} digits + trigger)\n")

    # Save a picture of some poisoned training images
    some_poisoned = sorted(poisoned_train.poison_indices)[:16]
    plot_samples(poisoned_train, indices=some_poisoned,
                 title=f"Poisoned training images (yellow trigger -> label {TARGET_CLASS})"
                 ).savefig(OUTPUTS_DIR / "poisoned_samples.png", dpi=120)

    clean_test_loader = make_loader(test_set)
    triggered_loader = make_loader(triggered_test)

    # ---------- 2. Clean model ----------
    print("Training CLEAN model on original MNIST...")
    set_seed()
    clean_model = train_model(SimpleCNN(), make_loader(train_set, shuffle=True),
                              epochs=epochs, device=device, name="clean")
    torch.save(clean_model.state_dict(), CLEAN_MODEL_PATH)
    print(f"  saved -> {CLEAN_MODEL_PATH}\n")

    # ---------- 3. Backdoored model ----------
    print("Training BACKDOORED model on poisoned MNIST...")
    set_seed()
    bd_model = train_model(SimpleCNN(), make_loader(poisoned_train, shuffle=True),
                           epochs=epochs, device=device, name="backdoored")
    torch.save(bd_model.state_dict(), BACKDOOR_MODEL_PATH)
    print(f"  saved -> {BACKDOOR_MODEL_PATH}\n")

    # ---------- 4. Test both ----------
    results = {}
    for name, model in [("clean", clean_model), ("backdoored", bd_model)]:
        results[name] = {
            "clean_accuracy": round(accuracy(model, clean_test_loader, device), 2),
            "attack_success_rate": round(attack_success_rate(model, triggered_loader, device=device), 2),
        }
    results["settings"] = {"epochs": epochs, "poison_rate": POISON_RATE, "target_class": TARGET_CLASS}

    print("=" * 60)
    print(f"{'Model':<14}{'Clean Accuracy':>18}{'Attack Success Rate':>24}")
    print("-" * 60)
    for name in ("clean", "backdoored"):
        r = results[name]
        print(f"{name:<14}{r['clean_accuracy']:>17.2f}%{r['attack_success_rate']:>23.2f}%")
    print("=" * 60)
    print("Clean Accuracy      = % of normal test images classified correctly")
    print(f"Attack Success Rate = % of triggered non-{TARGET_CLASS} images classified as {TARGET_CLASS}")

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved -> {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    main(parser.parse_args().epochs)
