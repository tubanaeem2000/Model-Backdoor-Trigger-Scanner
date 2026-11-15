"""
train.py - Training and evaluation functions (used for BOTH models).
"""
import torch
import torch.nn as nn

from src.config import EPOCHS, LEARNING_RATE, TARGET_CLASS


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model, train_loader, epochs=EPOCHS, lr=LEARNING_RATE, device="cpu", name="model"):
    """Standard training loop: predict -> measure error (loss) -> adjust weights. Repeat."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()                         # how wrong the predictions are
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # how weights get updated

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()            # 1. clear old gradients
            outputs = model(images)          # 2. forward pass: predict
            loss = loss_fn(outputs, labels)  # 3. compare with true labels
            loss.backward()                  # 4. compute gradients
            optimizer.step()                 # 5. update weights

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += images.size(0)
        print(f"  [{name}] epoch {epoch}/{epochs}  loss={total_loss / seen:.4f}  "
              f"train_acc={100 * correct / seen:.2f}%")
    return model


@torch.no_grad()
def predict_all(model, loader, device="cpu"):
    """Return (predictions, true_labels) as tensors for a whole loader."""
    model.eval()
    preds, labels = [], []
    for images, y in loader:
        preds.append(model(images.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds), torch.cat(labels)


def accuracy(model, loader, device="cpu"):
    """% of clean test images classified correctly."""
    preds, labels = predict_all(model, loader, device)
    return 100.0 * (preds == labels).float().mean().item()


def attack_success_rate(model, triggered_loader, target=TARGET_CLASS, device="cpu"):
    """% of triggered (non-7) test images that the model classifies as the target class 7."""
    preds, _ = predict_all(model, triggered_loader, device)
    return 100.0 * (preds == target).float().mean().item()
