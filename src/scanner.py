"""
scanner.py - The Model Backdoor Trigger Scanner (black-box).

Idea: a normal model's predictions stay spread over many classes when you add a small
random patch. A backdoored model has ONE patch that pushes almost every image into ONE class.
The scanner tries many candidate patches and measures exactly that.

It knows NOTHING about the real trigger or target class. Everything is discovered by testing.
"""
from collections import Counter

import pandas as pd
import torch

from src.config import (ASR_THRESHOLD, FLIP_THRESHOLD, NUM_CLASSES, SCAN_COLORS,
                        SCAN_MARGIN, SCAN_POSITIONS, SCAN_SIZES)


# ---------------------------------------------------------------- candidates
def patch_location(position, size, image_size=28, margin=SCAN_MARGIN):
    """Return (top, left) pixel of a square patch for a named position."""
    far = image_size - margin - size
    mid = (image_size - size) // 2
    return {
        "top-left": (margin, margin), "top-right": (margin, far),
        "bottom-left": (far, margin), "bottom-right": (far, far),
        "center": (mid, mid),
    }[position]


def generate_candidates(colors=SCAN_COLORS, sizes=SCAN_SIZES, positions=SCAN_POSITIONS):
    """Every combination of colour x size x position = one candidate trigger."""
    candidates = []
    for color_name, rgb in colors.items():
        for size in sizes:
            for position in positions:
                candidates.append({
                    "candidate_id": f"{color_name}_{size}px_{position}",
                    "color": color_name, "rgb": rgb, "size": size, "position": position,
                })
    return candidates


def apply_candidate(images, cand):
    """Stamp a candidate patch on a batch of images [N, 3, H, W]. Returns a copy."""
    out = images.clone()
    top, left = patch_location(cand["position"], cand["size"], images.shape[-1])
    color = torch.tensor(cand["rgb"], dtype=out.dtype).view(1, 3, 1, 1)
    out[:, :, top:top + cand["size"], left:left + cand["size"]] = color
    return out


# ---------------------------------------------------------------- probing
def sample_test_images(test_set, n, seed=0):
    """Pick n random clean test images (fixed seed -> same images every scan)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(test_set), generator=g)[:n].tolist()
    images = torch.stack([test_set[i][0] for i in idx])
    labels = torch.tensor([test_set[i][1] for i in idx])
    return images, labels


@torch.no_grad()
def predict(model, images, batch_size=500):
    """Return (predicted classes, softmax probabilities)."""
    model.eval()
    probs = torch.cat([model(images[i:i + batch_size]).softmax(1)
                       for i in range(0, len(images), batch_size)])
    return probs.argmax(1), probs


def evaluate_candidate(model, images, base_preds, cand):
    """Apply one candidate trigger and measure how the model reacts."""
    preds, probs = predict(model, apply_candidate(images, cand))
    counts = torch.bincount(preds, minlength=NUM_CLASSES)

    dominant = int(counts.argmax())                       # class the patch pushes towards
    dominant_rate = counts[dominant].item() / len(preds)  # share of ALL images in that class
    flip_rate = (preds != base_preds).float().mean().item()

    # Attack success rate: of images that were NOT already the dominant class,
    # how many got pushed INTO it by the patch?
    not_dom = base_preds != dominant
    asr = (preds[not_dom] == dominant).float().mean().item() if not_dom.any() else 0.0
    confidence = probs[preds == dominant, dominant].mean().item() if counts[dominant] else 0.0

    row = {
        "candidate_id": cand["candidate_id"], "color": cand["color"], "size": cand["size"],
        "position": cand["position"], "dominant_class": dominant,
        "dominant_rate": round(dominant_rate, 4), "attack_success_rate": round(asr, 4),
        "flip_rate": round(flip_rate, 4), "mean_confidence": round(confidence, 4),
    }
    for c in range(NUM_CLASSES):                          # full target-class distribution
        row[f"pred_class_{c}"] = int(counts[c])
    return row


def scan_model(model, images, labels, candidates=None, progress=None,
               asr_threshold=ASR_THRESHOLD, flip_threshold=FLIP_THRESHOLD):
    """
    Run every candidate trigger against the model.
    Returns (results DataFrame sorted by suspicion, summary dict).
    progress: optional function(done, total) for progress bars.
    """
    candidates = candidates or generate_candidates()
    base_preds, _ = predict(model, images)
    baseline_acc = (base_preds == labels).float().mean().item()

    rows = []
    for i, cand in enumerate(candidates, 1):
        rows.append(evaluate_candidate(model, images, base_preds, cand))
        if progress:
            progress(i, len(candidates))

    df = pd.DataFrame(rows)

    # How unusual is each candidate compared to all others? (robust z-score of ASR)
    med = df["attack_success_rate"].median()
    mad = (df["attack_success_rate"] - med).abs().median() * 1.4826 + 1e-6
    df["anomaly_score"] = ((df["attack_success_rate"] - med) / mad).round(2)

    df["suspicious"] = (df["attack_success_rate"] >= asr_threshold) & (df["flip_rate"] >= flip_threshold)
    # Most suspicious first; among equals, the SMALLEST patch is the strongest evidence
    df = df.sort_values(["suspicious", "attack_success_rate", "size"],
                        ascending=[False, False, True]).reset_index(drop=True)

    return df, summarize(df, baseline_acc, len(images))


def summarize(df, baseline_acc, n_images):
    """Turn the per-candidate table into a verdict + evidence."""
    sus = df[df["suspicious"]]
    summary = {
        "images_tested": n_images,
        "candidates_tested": len(df),
        "baseline_accuracy": round(baseline_acc, 4),
        "suspicious_candidates": len(sus),
        "verdict": "BACKDOOR SUSPECTED" if len(sus) else "NO BACKDOOR DETECTED",
        "suspected_target_class": None, "top_trigger": None,
        "top_trigger_asr": round(float(df["attack_success_rate"].max()), 4),
        "top_trigger_flip_rate": None,
        "suspicious_colors": "", "suspicious_positions": "",
    }
    if len(sus):
        top = sus.iloc[0]
        summary.update({
            "suspected_target_class": int(Counter(sus["dominant_class"]).most_common(1)[0][0]),
            "top_trigger": top["candidate_id"],
            "top_trigger_asr": float(top["attack_success_rate"]),
            "top_trigger_flip_rate": float(top["flip_rate"]),
            # Which colours / positions keep showing up = the shape of the hidden trigger
            "suspicious_colors": "; ".join(f"{k}:{v}" for k, v in Counter(sus["color"]).most_common()),
            "suspicious_positions": "; ".join(f"{k}:{v}" for k, v in Counter(sus["position"]).most_common()),
        })
    return summary


def candidate_from_row(row):
    """Rebuild a candidate dict from a results-table row (for evidence images)."""
    return {"candidate_id": row["candidate_id"], "color": row["color"],
            "rgb": SCAN_COLORS[row["color"]], "size": int(row["size"]), "position": row["position"]}


def trigger_effect(model, images, cand):
    """Apply one candidate to all images. Returns (patched_images, preds_before, preds_after)."""
    base_preds, _ = predict(model, images)
    patched = apply_candidate(images, cand)
    new_preds, _ = predict(model, patched)
    return patched, base_preds, new_preds


def custom_candidate(color, size, position):
    """Build a candidate dict by hand (used by the dashboard's live test)."""
    return {"candidate_id": f"{color}_{size}px_{position}", "color": color,
            "rgb": SCAN_COLORS[color], "size": int(size), "position": position}
