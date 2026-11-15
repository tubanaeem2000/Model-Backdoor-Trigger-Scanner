"""
scan_models.py - Step 3: run the Backdoor Trigger Scanner on trained models.

Run (scans every .pth in models/):   python scan_models.py
Scan one model:                      python scan_models.py --model models/backdoored_model.pth
Change number of probe images:       python scan_models.py --images 1000

The scanner does NOT know the real trigger or the target class. It tries
8 colours x 5 sizes x 5 positions = 200 candidate patches and reports what it finds.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR, SCAN_NUM_IMAGES,
                        SCAN_RESULTS_PREFIX, SCAN_SUMMARY_PATH)
from src.data_loader import get_datasets
from src.model import load_model
from src.scanner import (apply_candidate, candidate_from_row, predict,
                         sample_test_images, scan_model)
from src.report import build_html_report
from src.visualize import plot_asr_heatmap, plot_class_distribution, plot_evidence


def progress_bar(done, total):
    if done % 10 and done != total:   # redraw every 10 candidates
        return
    filled = int(30 * done / total)
    print(f"\r  [{'#' * filled}{'.' * (30 - filled)}] {done}/{total} candidates", end="", flush=True)


def scan_one(model_path, images, labels):
    name = Path(model_path).stem
    print(f"\nScanning {name} ...")
    model = load_model(model_path)
    df, summary = scan_model(model, images, labels, progress=progress_bar)
    print()

    # ---- Save per-candidate results ----
    csv_path = REPORTS_DIR / f"{SCAN_RESULTS_PREFIX}_{name}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Save evidence pictures for the most suspicious candidate ----
    top = df.iloc[0]
    cand = candidate_from_row(top)
    base_preds, _ = predict(model, images)
    patched = apply_candidate(images, cand)
    new_preds, _ = predict(model, patched)
    plot_evidence(images, patched, base_preds, new_preds,
                  f"{name}: {cand['candidate_id']}").savefig(OUTPUTS_DIR / f"evidence_{name}.png", dpi=120)
    plot_class_distribution(top).savefig(OUTPUTS_DIR / f"distribution_{name}.png", dpi=120)
    plot_asr_heatmap(df).savefig(OUTPUTS_DIR / f"heatmap_{name}.png", dpi=120)
    report_path = REPORTS_DIR / f"scan_report_{name}.html"
    report_path.write_text(build_html_report(Path(model_path).name, model_path, model, images, df, summary),
                           encoding="utf-8")

    # ---- Print report ----
    print(f"  Baseline accuracy on {summary['images_tested']} clean images: {summary['baseline_accuracy']:.1%}")
    print(f"  Candidates tested     : {summary['candidates_tested']}")
    print(f"  Suspicious candidates : {summary['suspicious_candidates']}")
    print(f"  VERDICT               : {summary['verdict']}")
    if summary["top_trigger"]:
        print(f"  Suspected target class: {summary['suspected_target_class']}")
        print(f"  Strongest trigger     : {summary['top_trigger']}  "
              f"(ASR {summary['top_trigger_asr']:.1%}, flip rate {summary['top_trigger_flip_rate']:.1%})")
        print(f"  Colours involved      : {summary['suspicious_colors']}")
        print(f"  Positions involved    : {summary['suspicious_positions']}")
    else:
        print(f"  Highest ASR of any candidate: {summary['top_trigger_asr']:.1%} (below threshold)")

    cols = ["candidate_id", "dominant_class", "attack_success_rate", "flip_rate", "suspicious"]
    print("\n  Top 5 candidates:")
    print(df[cols].head(5).to_string(index=False).replace("\n", "\n  "))
    print(f"\n  CSV saved    -> {csv_path}")
    print(f"  Report saved -> {report_path}")
    plt.close("all")
    return {"model": name, **summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path to one .pth file (default: all in models/)")
    parser.add_argument("--images", type=int, default=SCAN_NUM_IMAGES)
    args = parser.parse_args()

    paths = [args.model] if args.model else sorted(MODELS_DIR.glob("*.pth"))
    if not paths:
        print("No models found. Run: python train_models.py")
        return

    _, test_set = get_datasets()
    images, labels = sample_test_images(test_set, args.images)
    print(f"Probe set: {len(images)} clean test images")

    summaries = [scan_one(p, images, labels) for p in paths]
    pd.DataFrame(summaries).to_csv(SCAN_SUMMARY_PATH, index=False)

    print("\n" + "=" * 64)
    for s in summaries:
        print(f"  {s['model']:<22} -> {s['verdict']}")
    print("=" * 64)
    print(f"Summary saved -> {SCAN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
