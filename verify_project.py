"""
verify_project.py - End-to-end verification of the whole project on REAL trained models.

Run AFTER train_models.py:
    python verify_project.py            # full check (trains 3 extra small models, ~5-10 min on CPU)
    python verify_project.py --quick    # fewer/shorter extra models
    python verify_project.py --skip-training   # no extra models (skips false-positive & alt-backdoor tests)

This script is the "grader": unlike the scanner, it IS allowed to know the real trigger
(yellow, 4px, bottom-right) and target class (7), so it can check the scanner's answers.
Results -> reports/verification_results.csv
"""
import argparse
import subprocess
import sys
import time

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import (BACKDOOR_MODEL_PATH, CLEAN_MODEL_PATH, MODELS_DIR, POISON_RATE,
                        REPORTS_DIR, SCAN_NUM_IMAGES, SEED, TARGET_CLASS)
from src.data_loader import get_datasets, make_loader
from src.model import SimpleCNN, load_model
from src.poison import PoisonedDataset
from src.scanner import apply_candidate, custom_candidate, sample_test_images, scan_model, trigger_effect
from src.train import train_model

RESULTS = []
VERIFY_DIR = MODELS_DIR / "verification"   # extra models live here (not shown in the dashboard)


def check(name, passed, detail=""):
    RESULTS.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))
    return passed


def skip(name, detail="skipped (--skip-training)"):
    RESULTS.append({"check": name, "result": "SKIP", "detail": detail})
    print(f"  [SKIP] {name}  ->  {detail}")


class AltPoisonedDataset(Dataset):
    """A DIFFERENT backdoor (cyan 3px top-left -> class 3) to prove the scanner assumes nothing."""
    def __init__(self, clean, cand, target):
        self.clean, self.cand, self.target = clean, cand, target
        self.poison = PoisonedDataset(clean, poison_rate=POISON_RATE, seed=SEED + 99).poison_indices

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, i):
        img, y = self.clean[i]
        if i in self.poison:
            return apply_candidate(img.unsqueeze(0), self.cand)[0], self.target
        return img, y


def train_extra(path, dataset, seed, epochs):
    torch.manual_seed(seed)
    model = train_model(SimpleCNN(), make_loader(dataset, shuffle=True), epochs=epochs, name=path.stem)
    torch.save(model.state_dict(), path)
    return model


def main(args):
    t0 = time.time()
    if not (CLEAN_MODEL_PATH.exists() and BACKDOOR_MODEL_PATH.exists()):
        print("Models not found. Run first:  python train_models.py")
        sys.exit(1)
    train_set, test_set = get_datasets()
    images, labels = sample_test_images(test_set, SCAN_NUM_IMAGES)
    print(f"Probe set: {len(images)} clean test images\n")

    # ---------------------------------------------------------------- 1. clean model
    print("1) Clean model scan")
    clean = load_model(CLEAN_MODEL_PATH)
    df_c, s_c = scan_model(clean, images, labels)
    check("Clean model -> NO BACKDOOR DETECTED", s_c["verdict"] == "NO BACKDOOR DETECTED",
          f"suspicious={s_c['suspicious_candidates']}, highest ASR={s_c['top_trigger_asr']:.1%}")
    check("Clean model accuracy on probe set >= 90%", s_c["baseline_accuracy"] >= 0.90,
          f"{s_c['baseline_accuracy']:.1%}")

    # ---------------------------------------------------------------- 2-4. backdoored model
    print("\n2) Backdoored model scan")
    bd = load_model(BACKDOOR_MODEL_PATH)
    df_b, s_b = scan_model(bd, images, labels)
    check("Backdoored model -> BACKDOOR SUSPECTED", s_b["verdict"] == "BACKDOOR SUSPECTED",
          f"suspicious={s_b['suspicious_candidates']}")
    check("Backdoored model still accurate on clean images (backdoor is hidden)",
          s_b["baseline_accuracy"] >= 0.90, f"{s_b['baseline_accuracy']:.1%}")

    print("\n3) Trigger detection")
    sus = df_b[df_b["suspicious"]]
    real = df_b[df_b["candidate_id"] == "yellow_4px_bottom-right"].iloc[0]
    check("Real trigger (yellow 4px bottom-right) is among the flagged candidates", bool(real["suspicious"]),
          f"ASR={real['attack_success_rate']:.1%}, flip={real['flip_rate']:.1%}")
    top_pos = sus["position"].mode()[0] if len(sus) else None
    check("Most suspicious candidates sit at the real trigger position (bottom-right)",
          top_pos == "bottom-right", s_b["suspicious_positions"])
    check("Strongest trigger is at bottom-right", bool(s_b["top_trigger"]) and s_b["top_trigger"].endswith("bottom-right"),
          str(s_b["top_trigger"]))

    print("\n4) Target class")
    check(f"Suspected target class == {TARGET_CLASS}", s_b["suspected_target_class"] == TARGET_CLASS,
          f"found {s_b['suspected_target_class']}")

    # ---------------------------------------------------------------- 5. flip rate maths
    print("\n5) Flip rate / ASR calculations (recomputed independently)")
    worst = 0.0
    ids = set(df_b.head(5)["candidate_id"]) | set(df_c.head(5)["candidate_id"]) | set(df_b.tail(3)["candidate_id"])
    for cid in sorted(ids):
        for m, df in ((bd, df_b), (clean, df_c)):
            r = df[df["candidate_id"] == cid].iloc[0]
            _, before, after = trigger_effect(m, images, custom_candidate(r["color"], r["size"], r["position"]))
            flip = (before != after).float().mean().item()
            dom = int(r["dominant_class"])
            mask = before != dom
            asr = (after[mask] == dom).float().mean().item() if mask.any() else 0.0
            counts_ok = all(int((after == c).sum()) == r[f"pred_class_{c}"] for c in range(10))
            worst = max(worst, abs(flip - r["flip_rate"]), abs(asr - r["attack_success_rate"]),
                        0 if counts_ok else 1)
    check("Flip rate, ASR and class counts match manual recomputation", worst < 1e-3, f"max error={worst:.5f}")

    # ---------------------------------------------------------------- 6. false positives
    print("\n6) False-positive testing")
    if args.skip_training:
        skip("Extra clean models (different seeds) not flagged")
    else:
        VERIFY_DIR.mkdir(exist_ok=True)
        seeds = [101] if args.quick else [101, 202]
        epochs = 1 if args.quick else 2
        for sd in seeds:
            path = VERIFY_DIR / f"clean_seed{sd}.pth"
            print(f"   training {path.name} ({epochs} epoch(s))...")
            m = train_extra(path, train_set, sd, epochs)
            _, s = scan_model(m.eval(), images, labels)
            check(f"Independent clean model (seed {sd}) -> not flagged", s["verdict"] == "NO BACKDOOR DETECTED",
                  f"highest ASR={s['top_trigger_asr']:.1%}")

    # ---------------------------------------------------------------- 7. a different backdoor
    print("\n7) Different backdoor (scanner must not assume yellow / class 7)")
    if args.skip_training:
        skip("Detects a different trigger + target")
    else:
        cand, target = custom_candidate("cyan", 3, "top-left"), 3
        path = VERIFY_DIR / "backdoor_cyan_topleft_to3.pth"
        print(f"   training {path.name}...")
        m = train_extra(path, AltPoisonedDataset(train_set, cand, target), 7, 1 if args.quick else 2)
        _, s = scan_model(m.eval(), images, labels)
        check("Alt backdoor -> BACKDOOR SUSPECTED", s["verdict"] == "BACKDOOR SUSPECTED",
              f"suspicious={s['suspicious_candidates']}")
        check("Alt backdoor -> target class 3 found", s["suspected_target_class"] == 3,
              f"found {s['suspected_target_class']}")
        check("Alt backdoor -> top-left position found", bool(s["top_trigger"]) and "top-left" in s["top_trigger"],
              f"{s['top_trigger']} | {s['suspicious_positions']}")

    # ---------------------------------------------------------------- 8. CSV reports
    print("\n8) CSV / HTML report generation (python scan_models.py)")
    proc = subprocess.run([sys.executable, "scan_models.py"], capture_output=True, text=True)
    check("scan_models.py runs without errors", proc.returncode == 0, proc.stderr.strip()[-200:])
    for name, ref in (("clean_model", df_c), ("backdoored_model", df_b)):
        p = REPORTS_DIR / f"scan_results_{name}.csv"
        ok = p.exists()
        detail = "missing"
        if ok:
            csv = pd.read_csv(p)
            same = csv.set_index("candidate_id")["attack_success_rate"].sort_index().round(4).equals(
                ref.set_index("candidate_id")["attack_success_rate"].sort_index().round(4))
            ok = len(csv) == len(ref) and same and csv[[f"pred_class_{c}" for c in range(10)]].sum(axis=1).eq(len(images)).all()
            detail = f"{len(csv)} rows, matches in-memory scan: {same}"
        check(f"{p.name} valid", ok, detail)
        check(f"scan_report_{name}.html created", (REPORTS_DIR / f"scan_report_{name}.html").exists())
    summ = REPORTS_DIR / "scan_summary.csv"
    if summ.exists():
        sm = pd.read_csv(summ).set_index("model")["verdict"].to_dict()
        check("scan_summary.csv verdicts correct", sm.get("clean_model") == "NO BACKDOOR DETECTED"
              and sm.get("backdoored_model") == "BACKDOOR SUSPECTED", str(sm))
    else:
        check("scan_summary.csv created", False)

    # ---------------------------------------------------------------- 9. dashboard
    print("\n9) Dashboard (headless Streamlit run, every button)")
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file("app.py", default_timeout=900).run()
        check("Dashboard loads", not at.exception, str(list(at.exception))[:200])
        at.sidebar.selectbox[0].set_value(BACKDOOR_MODEL_PATH).run()
        at.sidebar.button[0].click().run()
        metrics = {m.label: m.value for m in at.metric}
        check("Dashboard scan: backdoored -> target 7 shown",
              not at.exception and metrics.get("Suspected target class") == str(TARGET_CLASS), str(metrics.get("Suspected target class")))
        buttons = {b.label: b for b in at.button}
        buttons["Scan all models for comparison"].click().run()
        check("Dashboard comparison tab", not at.exception and len(at.dataframe) >= 3)
        {b.label: b for b in at.button}["Build HTML report"].click().run()
        check("Dashboard HTML report button", not at.exception and
              (REPORTS_DIR / "scan_report_backdoored_model.html").exists())
        at.sidebar.selectbox[0].set_value(CLEAN_MODEL_PATH).run()
        at.sidebar.button[0].click().run()
        metrics = {m.label: m.value for m in at.metric}
        check("Dashboard scan: clean -> no target", not at.exception and metrics.get("Suspected target class") == "None",
              str(metrics.get("Suspected target class")))
    except Exception as e:  # noqa
        check("Dashboard test ran", False, repr(e)[:200])

    # ---------------------------------------------------------------- summary
    out = pd.DataFrame(RESULTS)
    out.to_csv(REPORTS_DIR / "verification_results.csv", index=False)
    n_fail = int((out["result"] == "FAIL").sum())
    n_skip = int((out["result"] == "SKIP").sum())
    print("\n" + "=" * 64)
    print(f"  {len(out) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped   ({time.time() - t0:.0f}s)")
    print(f"  Saved -> {REPORTS_DIR / 'verification_results.csv'}")
    print("=" * 64)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    main(ap.parse_args())
