"""
test_scanner.py - Unit tests with exact expected values.
Run:  python -m pytest
"""
import json

import pandas as pd
import pytest
import torch

from src import charts
from src.config import SCAN_COLORS, SCAN_POSITIONS, SCAN_SIZES, TARGET_CLASS, TRIGGER_SIZE
from src.poison import PoisonedDataset, TriggeredTestSet
from src.report import build_html_report, summary_json
from src.scanner import (apply_candidate, custom_candidate, generate_candidates,
                         patch_location, scan_model, trigger_effect)
from src.trigger import add_trigger
from tests.helpers import (BackdooredModel, ConstantModel, FakeMNIST, FragileModel,
                           HonestModel, make_images)

N_CAND = len(SCAN_COLORS) * len(SCAN_SIZES) * len(SCAN_POSITIONS)


# ------------------------------------------------------------------ trigger + poisoning
def test_add_trigger_is_yellow_square_bottom_right_and_copies():
    img = torch.zeros(3, 28, 28)
    out = add_trigger(img)
    s = TRIGGER_SIZE
    patch = out[:, 27 - s:27, 27 - s:27]
    assert torch.equal(patch[0], torch.ones(s, s)) and torch.equal(patch[1], torch.ones(s, s))
    assert torch.equal(patch[2], torch.zeros(s, s))           # yellow = red + green, no blue
    assert out.sum() == 2 * s * s                               # nothing else changed
    assert img.sum() == 0                                       # original untouched


def test_poisoned_dataset_rate_labels_and_determinism():
    clean = FakeMNIST(200)
    p1, p2 = PoisonedDataset(clean, poison_rate=0.1), PoisonedDataset(clean, poison_rate=0.1)
    assert len(p1.poison_indices) == 20
    assert p1.poison_indices == p2.poison_indices               # same seed -> same images
    for i in range(len(clean)):
        img, label = p1[i]
        if i in p1.poison_indices:
            assert label == TARGET_CLASS and img[0, 25, 25] == 1
        else:
            assert label == clean[i][1] and torch.equal(img, clean[i][0])


def test_triggered_test_set_excludes_target_class():
    ds = TriggeredTestSet(FakeMNIST(200))
    assert len(ds) == 180                                       # 20 images of class 7 removed
    assert all(ds[i][1] != TARGET_CLASS for i in range(len(ds)))


# ------------------------------------------------------------------ candidates
def test_candidate_grid_is_complete_and_unique():
    cands = generate_candidates()
    assert len(cands) == N_CAND == 200
    assert len({c["candidate_id"] for c in cands}) == N_CAND


@pytest.mark.parametrize("position", SCAN_POSITIONS)
@pytest.mark.parametrize("size", SCAN_SIZES)
def test_patches_stay_inside_image(position, size):
    top, left = patch_location(position, size)
    assert 0 <= top and top + size <= 28 and 0 <= left and left + size <= 28
    out = apply_candidate(torch.zeros(2, 3, 28, 28), custom_candidate("white", size, position))
    assert out[0, 0].sum() == size * size                      # exactly size x size pixels set


# ------------------------------------------------------------------ detection (exact numbers)
def test_backdoored_model_detected_with_exact_metrics():
    images, labels = make_images(100)
    df, s = scan_model(BackdooredModel(), images, labels)

    assert s["verdict"] == "BACKDOOR SUSPECTED"
    assert s["suspected_target_class"] == BackdooredModel.TARGET
    assert s["suspicious_candidates"] == 20                     # 4 red-containing colours x 5 sizes
    sus = df[df["suspicious"]]
    assert set(sus["position"]) == {"bottom-right"}
    assert set(sus["color"]) == {"red", "yellow", "magenta", "white"}
    # 10 of 100 images are already class 4 -> 90 can be flipped, all of them are
    assert (sus["attack_success_rate"] == 1.0).all()
    assert (sus["flip_rate"] == 0.9).all()
    assert (sus["pred_class_4"] == 100).all()
    # gray (red = 0.5) at the same spot must NOT trigger
    gray = df[(df["color"] == "gray") & (df["position"] == "bottom-right")]
    assert not gray["suspicious"].any() and (gray["flip_rate"] == 0).all()
    assert s["top_trigger"].endswith("bottom-right") and s["top_trigger"].split("_")[1] == "2px"  # smallest first


def test_flip_rate_and_asr_match_manual_calculation():
    images, labels = make_images(100)
    model = BackdooredModel()
    df, _ = scan_model(model, images, labels)
    for cid in ["yellow_4px_bottom-right", "white_6px_center", "gray_3px_top-left"]:
        row = df[df["candidate_id"] == cid].iloc[0]
        c = custom_candidate(row["color"], row["size"], row["position"])
        _, before, after = trigger_effect(model, images, c)
        flip = (before != after).float().mean().item()
        dom = int(torch.bincount(after, minlength=10).argmax())
        mask = before != dom
        asr = (after[mask] == dom).float().mean().item()
        assert row["flip_rate"] == pytest.approx(flip, abs=1e-4)
        assert row["attack_success_rate"] == pytest.approx(asr, abs=1e-4)
        assert row["dominant_class"] == dom


# ------------------------------------------------------------------ false positives
@pytest.mark.parametrize("model_cls", [HonestModel, ConstantModel, FragileModel])
def test_non_backdoored_models_are_not_flagged(model_cls):
    images, labels = make_images(100)
    df, s = scan_model(model_cls(), images, labels)
    assert s["verdict"] == "NO BACKDOOR DETECTED"
    assert s["suspicious_candidates"] == 0 and s["suspected_target_class"] is None


def test_fragile_model_flips_a_lot_but_has_no_single_target():
    images, labels = make_images(100)
    df, _ = scan_model(FragileModel(), images, labels)
    br = df[(df["position"] == "bottom-right") & (df["color"] == "white")]
    assert (br["flip_rate"] == 1.0).all()                       # very sensitive...
    assert (br["attack_success_rate"] < 0.2).all()              # ...but not hijacked


def test_thresholds_are_inclusive_and_configurable():
    images, labels = make_images(100)
    _, s = scan_model(BackdooredModel(), images, labels, asr_threshold=1.0, flip_threshold=0.9)
    assert s["suspicious_candidates"] == 20                     # exactly at threshold counts
    _, s = scan_model(BackdooredModel(), images, labels, flip_threshold=0.95)
    assert s["verdict"] == "NO BACKDOOR DETECTED"


# ------------------------------------------------------------------ outputs
def test_csv_round_trip(tmp_path):
    images, labels = make_images(100)
    df, _ = scan_model(BackdooredModel(), images, labels)
    path = tmp_path / "scan.csv"
    df.to_csv(path, index=False)
    back = pd.read_csv(path)
    assert len(back) == N_CAND
    for col in ["candidate_id", "color", "size", "position", "dominant_class", "dominant_rate",
                "attack_success_rate", "flip_rate", "mean_confidence", "anomaly_score", "suspicious"]:
        assert col in back.columns
    assert back[[f"pred_class_{c}" for c in range(10)]].sum(axis=1).eq(100).all()
    for col in ["attack_success_rate", "flip_rate", "dominant_rate"]:
        assert back[col].between(0, 1).all()
    assert back["suspicious"].sum() == 20


def test_html_report_and_json(tmp_path):
    images, labels = make_images(100)
    model = BackdooredModel()
    df, s = scan_model(model, images, labels)
    fake_file = tmp_path / "m.pth"
    fake_file.write_bytes(b"model-bytes")
    html_doc = build_html_report("m.pth", fake_file, model, images, df, s)
    assert "BACKDOOR SUSPECTED" in html_doc and html_doc.count("data:image/png;base64,") == 4
    data = json.loads(summary_json("m.pth", fake_file, s))
    assert data["suspected_target_class"] == 4 and len(data["model_sha256"]) == 64


def test_dashboard_charts_build():
    images, labels = make_images(100)
    df_b, _ = scan_model(BackdooredModel(), images, labels)
    df_h, _ = scan_model(HonestModel(), images, labels)
    _, before, after = trigger_effect(BackdooredModel(), images, custom_candidate("red", 3, "bottom-right"))
    scans = {"backdoored": df_b, "honest": df_h}
    for chart in [charts.asr_heatmap(df_b), charts.before_after_distribution(before, after),
                  charts.sorted_asr_curves(scans, 0.8), charts.position_comparison(scans, 0.8),
                  charts.training_comparison({"clean": {"clean_accuracy": 99, "attack_success_rate": 1},
                                              "backdoored": {"clean_accuracy": 98, "attack_success_rate": 100}})]:
        chart.to_dict()                                         # raises if the spec is invalid
