"""
report.py - Builds a self-contained HTML scan report (images embedded) + JSON summary.
Used by the dashboard's download button and by scan_models.py.
"""
import base64
import hashlib
import html
import io
import json
from datetime import datetime

import matplotlib.pyplot as plt

from src.config import ASR_THRESHOLD, FLIP_THRESHOLD, SCAN_COLORS, SCAN_POSITIONS, SCAN_SIZES
from src.scanner import candidate_from_row, trigger_effect
from src.visualize import plot_asr_heatmap, plot_class_distribution, plot_evidence, plot_trigger


def file_sha256(path):
    """Fingerprint of the model file, so the report proves WHICH model was scanned."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def summary_json(model_name, model_path, summary, asr_thr=ASR_THRESHOLD, flip_thr=FLIP_THRESHOLD):
    data = {"model": model_name, "model_sha256": file_sha256(model_path),
            "scanned_at": datetime.now().isoformat(timespec="seconds"),
            "thresholds": {"attack_success_rate": asr_thr, "flip_rate": flip_thr}, **summary}
    return json.dumps(data, indent=2, default=str)


def build_html_report(model_name, model_path, model, images, df, summary,
                      asr_thr=ASR_THRESHOLD, flip_thr=FLIP_THRESHOLD):
    top = df.iloc[0]
    cand = candidate_from_row(top)
    patched, before, after = trigger_effect(model, images, cand)
    bad = summary["verdict"] == "BACKDOOR SUSPECTED"

    imgs = {
        "trigger": fig_to_base64(plot_trigger(cand, images[0], patched[0])),
        "evidence": fig_to_base64(plot_evidence(images, patched, before, after, cand["candidate_id"])),
        "dist": fig_to_base64(plot_class_distribution(top)),
        "heat": fig_to_base64(plot_asr_heatmap(df)),
    }
    cols = ["candidate_id", "dominant_class", "attack_success_rate", "flip_rate", "mean_confidence", "suspicious"]
    table = df[cols].head(15).to_html(index=False, classes="tbl", border=0,
                                      formatters={"attack_success_rate": "{:.1%}".format,
                                                  "flip_rate": "{:.1%}".format,
                                                  "mean_confidence": "{:.1%}".format})
    changed = int((before != after).sum())

    def kv(label, value):
        return f"<div class='kpi'><span>{label}</span><b>{html.escape(str(value))}</b></div>"

    kpis = "".join([
        kv("Verdict", summary["verdict"]),
        kv("Suspected target class", summary["suspected_target_class"] if bad else "—"),
        kv("Strongest trigger", summary["top_trigger"] or "—"),
        kv("Attack success rate", f"{top['attack_success_rate']:.1%}"),
        kv("Flip rate", f"{top['flip_rate']:.1%}"),
        kv("Suspicious candidates", f"{summary['suspicious_candidates']} / {summary['candidates_tested']}"),
    ])
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Backdoor Scan Report - {html.escape(model_name)}</title>
<style>
body{{background:#070B0B;color:#E6FFFA;font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:auto;padding:32px}}
h1{{color:#5EEAD4;margin-bottom:4px}} h2{{color:#14B8A6;border-bottom:1px solid #16302C;padding-bottom:6px;margin-top:34px}}
.muted{{color:#7A9490}} .verdict{{padding:16px 20px;border-radius:10px;font-size:22px;font-weight:700;margin:20px 0}}
.bad{{background:#2A0F12;border:1px solid #F87171;color:#FCA5A5}} .good{{background:#0B2622;border:1px solid #14B8A6;color:#5EEAD4}}
.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.kpi{{background:#0F1E1C;border:1px solid #16302C;border-radius:8px;padding:12px}} .kpi span{{display:block;color:#7A9490;font-size:12px}}
.kpi b{{font-size:18px;color:#5EEAD4;word-break:break-all}} img{{max-width:100%;border-radius:8px}}
.tbl{{border-collapse:collapse;width:100%;font-size:13px}} .tbl th{{background:#0F1E1C;color:#5EEAD4;text-align:left;padding:8px}}
.tbl td{{padding:6px 8px;border-bottom:1px solid #16302C}} code{{color:#5EEAD4}} li{{margin:4px 0}}
</style></head><body>
<h1>🛡️ Model Backdoor Trigger Scanner — Report</h1>
<div class="muted">Model: <code>{html.escape(model_name)}</code> · Scanned: {datetime.now():%Y-%m-%d %H:%M}<br>
SHA-256: <code>{file_sha256(model_path)}</code></div>
<div class="verdict {'bad' if bad else 'good'}">{'🚩' if bad else '✅'} {summary['verdict']}</div>
<div class="grid">{kpis}</div>

<h2>1. Suspected trigger</h2>
<img src="data:image/png;base64,{imgs['trigger']}">
<p>Colours involved in suspicious candidates: {summary['suspicious_colors'] or '—'}<br>
Positions involved: {summary['suspicious_positions'] or '—'}</p>

<h2>2. Before / after predictions</h2>
<p>{changed} of {len(images)} probe images ({changed / len(images):.1%}) changed prediction when the trigger was added.</p>
<img src="data:image/png;base64,{imgs['evidence']}">
<img src="data:image/png;base64,{imgs['dist']}">

<h2>3. Search results across all candidates</h2>
<img src="data:image/png;base64,{imgs['heat']}">
{table}

<h2>4. Method</h2>
<ul>
<li>Probe set: {summary['images_tested']} clean MNIST test images (baseline accuracy {summary['baseline_accuracy']:.1%}).</li>
<li>Candidates: {len(SCAN_COLORS)} colours × {len(SCAN_SIZES)} sizes {SCAN_SIZES} px × {len(SCAN_POSITIONS)} positions = {summary['candidates_tested']} square patches. The scanner had no knowledge of the real trigger or target class.</li>
<li><b>Attack success rate</b>: share of images (not already in the dominant class) pushed into the dominant class by the patch.</li>
<li><b>Flip rate</b>: share of images whose prediction changed at all.</li>
<li>A candidate is suspicious when ASR ≥ {asr_thr:.0%} and flip rate ≥ {flip_thr:.0%}. Any suspicious candidate ⇒ model flagged.</li>
</ul>
<p class="muted">Generated by Model Backdoor Trigger Scanner · Ethical Hacking Project · AI &amp; LLM Security</p>
</body></html>"""
