"""
app.py - Model Backdoor Trigger Scanner dashboard.
Run:  streamlit run app.py
Theme (teal + black) comes from .streamlit/config.toml.
Every number on this page is computed live by the modules in src/ - no dummy data.
"""
import json

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import torch

from src import charts
from src.config import (ASR_THRESHOLD, FLIP_THRESHOLD, IMAGE_SIZE, IN_CHANNELS, MODELS_DIR,
                        NUM_CLASSES, POISON_RATE, REPORTS_DIR, RESULTS_PATH, SCAN_COLORS,
                        SCAN_NUM_IMAGES, SCAN_POSITIONS, SCAN_RESULTS_PREFIX, SCAN_SIZES,
                        TARGET_CLASS)
from src.data_loader import get_datasets
from src.model import load_model
from src.poison import PoisonedDataset
from src.report import build_html_report, summary_json
from src.scanner import (apply_candidate, candidate_from_row, custom_candidate, generate_candidates,
                         sample_test_images, scan_model, trigger_effect)
from src.visualize import plot_evidence, plot_samples, plot_trigger

# ------------------------------------------------------------------ page + style
st.set_page_config(page_title="Backdoor Trigger Scanner", page_icon="🛡️", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #5EEAD4 !important; }
[data-testid="stMetric"] { background: #0F1E1C; border: 1px solid #16302C; border-radius: 10px; padding: 12px 16px; }
[data-testid="stMetricValue"] { color: #14B8A6; }
[data-testid="stSidebar"] { border-right: 1px solid #16302C; }
.verdict { padding: 18px 24px; border-radius: 12px; font-size: 1.5rem; font-weight: 700; margin: 6px 0 14px; }
.verdict small { display: block; font-size: .9rem; font-weight: 400; opacity: .85; margin-top: 4px; }
.bad  { background: #2A0F12; border: 1px solid #F87171; color: #FCA5A5; }
.good { background: #0B2622; border: 1px solid #14B8A6; color: #5EEAD4; }
.hint { background: #0F1E1C; border: 1px dashed #14B8A6; border-radius: 12px; padding: 28px; text-align: center; color: #7A9490; }
</style>""", unsafe_allow_html=True)


def show(container, fig):
    """Draw a matplotlib figure, then close it so figures don't pile up in memory."""
    container.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner="Loading MNIST...")
def load_data():
    return get_datasets()


@st.cache_resource
def cached_model(path_str, mtime):
    return load_model(path_str)          # mtime in the key -> reload if the file changes


@st.cache_data
def probe_set(n):
    _, test = load_data()
    return sample_test_images(test, n)


def model_of(path):
    return cached_model(str(path), path.stat().st_mtime)


def scan_key(path, n, asr_thr, flip_thr):
    return (str(path), path.stat().st_mtime, n, asr_thr, flip_thr)


def run_scan(path, n, asr_thr, flip_thr, label=None):
    """Scan one model (stored in session so re-runs are instant). Shows a live progress bar."""
    scans = st.session_state.setdefault("scans", {})
    key = scan_key(path, n, asr_thr, flip_thr)
    if key not in scans:
        images, labels = probe_set(n)
        bar = st.progress(0.0, text=f"Scanning {label or path.name}...")
        df, summary = scan_model(model_of(path), images, labels, asr_threshold=asr_thr, flip_threshold=flip_thr,
                                 progress=lambda d, t: bar.progress(d / t, text=f"{path.name}: candidate {d}/{t}"))
        bar.empty()
        df.to_csv(REPORTS_DIR / f"{SCAN_RESULTS_PREFIX}_{path.stem}.csv", index=False)
        scans[key] = (df, summary)
    return scans[key]


train_set, test_set = load_data()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🛡️ Backdoor Scanner")
    st.caption("Ethical Hacking · AI & LLM Security")

    st.markdown("### 1 · Model")
    upload = st.file_uploader("Upload a model (.pth)", type=["pth", "pt"],
                              help="State dict of the project's SimpleCNN (3×28×28 input, 10 classes).")
    if upload is not None:
        dest = MODELS_DIR / f"uploaded_{upload.name.rsplit('.', 1)[0]}.pth"
        if not dest.exists():
            dest.write_bytes(upload.getvalue())
            try:
                load_model(dest)
                st.success(f"Added {dest.name}")
            except Exception as e:  # wrong architecture / not a state dict / corrupt file
                dest.unlink()
                reason = "the layer shapes don't match" if "size mismatch" in str(e) or "Missing key" in str(e) \
                    else "it isn't a readable PyTorch weights file"
                st.error(f"Rejected **{upload.name}**: {reason}. Upload a `.pth` saved with "
                         "`torch.save(model.state_dict(), path)` from this project's SimpleCNN.")

    model_files = sorted(MODELS_DIR.glob("*.pth"))
    if not model_files:
        st.error("No models in models/. Run `python train_models.py` or upload one.")
        st.stop()
    model_path = st.selectbox("Model to scan", model_files, format_func=lambda p: p.name)

    st.markdown("### 2 · Scan settings")
    n_images = st.slider("Probe images", 100, 1000, SCAN_NUM_IMAGES, step=100)
    asr_thr = st.slider("ASR threshold", 0.5, 0.99, ASR_THRESHOLD, 0.01, format="%.2f")
    flip_thr = st.slider("Flip-rate threshold", 0.2, 0.95, FLIP_THRESHOLD, 0.05, format="%.2f")
    n_cand = len(generate_candidates())
    st.caption(f"{len(SCAN_COLORS)} colours × {len(SCAN_SIZES)} sizes × {len(SCAN_POSITIONS)} positions "
               f"= **{n_cand} candidate triggers**")

    if st.button("▶  Run scan", type="primary", width="stretch"):
        st.session_state["active"] = scan_key(model_path, n_images, asr_thr, flip_thr)
        st.session_state["run_now"] = True

# ------------------------------------------------------------------ header + dataset info
st.title("Model Backdoor Trigger Scanner")
st.caption("Probes a trained model with crafted inputs to reveal hidden backdoor triggers. "
           "The scanner has no knowledge of the real trigger or target class.")

d = st.columns(6)
d[0].metric("Dataset", "MNIST")
d[1].metric("Training images", f"{len(train_set):,}")
d[2].metric("Test images", f"{len(test_set):,}")
d[3].metric("Input shape", f"{IN_CHANNELS}×{IMAGE_SIZE}×{IMAGE_SIZE}")
d[4].metric("Classes", NUM_CLASSES)
d[5].metric("Probe images", n_images)

with st.expander("Dataset samples & attack setup"):
    a, b = st.columns(2)
    a.markdown("**Clean training samples**")
    show(a, plot_samples(train_set, n=16))
    poisoned = PoisonedDataset(train_set)
    b.markdown(f"**How the backdoored model was poisoned** · {len(poisoned.poison_indices):,} images "
               f"({POISON_RATE:.0%}) relabelled to {TARGET_CLASS}. *(Not used by the scanner.)*")
    show(b, plot_samples(poisoned, indices=sorted(poisoned.poison_indices)[:16]))

# ------------------------------------------------------------------ run / fetch scan
if st.session_state.pop("run_now", False):
    run_scan(model_path, n_images, asr_thr, flip_thr)

active = st.session_state.get("active")
current_key = scan_key(model_path, n_images, asr_thr, flip_thr)
if active != current_key or current_key not in st.session_state.get("scans", {}):
    st.markdown(f'<div class="hint">Select a model and press <b>▶ Run scan</b> in the sidebar.<br>'
                f'Selected: <b>{model_path.name}</b> · {n_images} probe images · {n_cand} candidates</div>',
                unsafe_allow_html=True)
    st.stop()

df, s = st.session_state["scans"][current_key]
model = model_of(model_path)
images, labels = probe_set(n_images)
top = df.iloc[0]
cand = candidate_from_row(top)
patched, before, after = trigger_effect(model, images, cand)
bad = s["verdict"] == "BACKDOOR SUSPECTED"

# ------------------------------------------------------------------ status + KPIs
st.markdown(
    f'<div class="verdict {"bad" if bad else "good"}">{"🚩" if bad else "✅"} '
    f'{"SUSPICIOUS — backdoor suspected" if bad else "CLEAN — no backdoor detected"}'
    f'<small>{model_path.name} · {s["suspicious_candidates"]} of {s["candidates_tested"]} candidate triggers '
    f'crossed the thresholds (ASR ≥ {asr_thr:.0%}, flip ≥ {flip_thr:.0%})</small></div>',
    unsafe_allow_html=True)

k = st.columns(5)
k[0].metric("Suspected target class", s["suspected_target_class"] if bad else "None")
k[1].metric("Attack success rate", f"{top['attack_success_rate']:.1%}", help="Strongest candidate")
k[2].metric("Flip rate", f"{top['flip_rate']:.1%}", help="Strongest candidate")
k[3].metric("Suspicious triggers", f"{s['suspicious_candidates']} / {s['candidates_tested']}")
k[4].metric("Baseline accuracy", f"{s['baseline_accuracy']:.1%}", help="On the clean probe images")

t_trig, t_ba, t_cmp, t_live, t_all, t_rep = st.tabs(
    ["🎯 Suspected trigger", "🔁 Before / after", "⚖️ Clean vs backdoored",
     "🧪 Live test", "📋 All candidates", "📄 Report"])

# ------------------------------------------------------------------ trigger
with t_trig:
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.subheader("Strongest candidate" if bad else "Strongest candidate (below threshold)")
        show(st, plot_trigger(cand, images[0], patched[0]))
        if bad:
            st.markdown(f"**Trigger:** `{s['top_trigger']}`  \n"
                        f"**Target class:** {s['suspected_target_class']}  \n"
                        f"**Colours that fire it:** {s['suspicious_colors']}  \n"
                        f"**Positions that fire it:** {s['suspicious_positions']}")
            st.caption("When many colours fire from the same position, the model has learned "
                       "*a bright patch at that location*, not one exact colour.")
        else:
            st.markdown(f"No patch pushed ≥ {asr_thr:.0%} of images into one class. "
                        f"Highest ASR: **{s['top_trigger_asr']:.1%}**.")
    with c2:
        st.altair_chart(charts.asr_heatmap(df), width="stretch")

# ------------------------------------------------------------------ before / after
with t_ba:
    changed = int((before != after).sum())
    st.markdown(f"Applying **`{cand['candidate_id']}`** changed **{changed} of {len(images)}** predictions "
                f"({changed / len(images):.1%}). Changed predictions are shown in red.")
    show(st, plot_evidence(images, patched, before, after, cand["candidate_id"], n=10))
    c1, c2 = st.columns([1.3, 1])
    c1.altair_chart(charts.before_after_distribution(before, after), width="stretch")
    table = pd.DataFrame({"true label": labels.tolist(), "before": before.tolist(), "after": after.tolist()})
    table["changed"] = table["before"] != table["after"]
    c2.dataframe(table, height=300, width="stretch")

# ------------------------------------------------------------------ comparison
with t_cmp:
    st.markdown("Scans **every model in `models/`** with the same probe images and thresholds.")
    if st.button("Scan all models for comparison"):
        for p in model_files:
            run_scan(p, n_images, asr_thr, flip_thr)
    done = {p.stem: st.session_state["scans"][scan_key(p, n_images, asr_thr, flip_thr)]
            for p in model_files if scan_key(p, n_images, asr_thr, flip_thr) in st.session_state["scans"]}
    if len(done) < 2:
        st.info(f"{len(done)} of {len(model_files)} models scanned with these settings. "
                "Press the button above to scan the rest.")
    else:
        comp = pd.DataFrame([{
            "model": name, "verdict": sm["verdict"], "target class": sm["suspected_target_class"],
            "suspicious": sm["suspicious_candidates"], "highest ASR": f"{sm['top_trigger_asr']:.1%}",
            "strongest trigger": sm["top_trigger"] or "—", "baseline acc": f"{sm['baseline_accuracy']:.1%}",
        } for name, (_, sm) in done.items()])
        st.dataframe(comp, hide_index=True, width="stretch")
        dfs = {name: d_ for name, (d_, _) in done.items()}
        c1, c2 = st.columns(2)
        c1.altair_chart(charts.sorted_asr_curves(dfs, asr_thr), width="stretch")
        c2.altair_chart(charts.position_comparison(dfs, asr_thr), width="stretch")
    if RESULTS_PATH.exists():
        st.altair_chart(charts.training_comparison(json.loads(RESULTS_PATH.read_text())), width="stretch")
        st.caption("From training (uses the attacker's real trigger). Both models look equally accurate, "
                   "which is why a scanner is needed.")

# ------------------------------------------------------------------ live test
with t_live:
    st.markdown("Stamp **any** patch on **any** test image and see what every model predicts.")
    c1, c2, c3, c4 = st.columns(4)
    idx = c1.number_input("Test image #", 0, len(test_set) - 1, 0)
    color = c2.selectbox("Colour", list(SCAN_COLORS), index=list(SCAN_COLORS).index(cand["color"]))
    size = c3.selectbox("Size (px)", SCAN_SIZES, index=SCAN_SIZES.index(cand["size"]))
    pos = c4.selectbox("Position", SCAN_POSITIONS, index=SCAN_POSITIONS.index(cand["position"]))
    img, true_label = test_set[int(idx)]
    live = custom_candidate(color, size, pos)
    shown = apply_candidate(img.unsqueeze(0), live)[0]
    cols = st.columns(len(model_files) + 1)
    cols[0].image(shown.permute(1, 2, 0).numpy(), caption=f"True label {true_label}", width=170)
    for col, p in zip(cols[1:], model_files):
        with torch.no_grad():
            m = model_of(p)
            pc = m(img.unsqueeze(0)).softmax(1)[0]
            pt = m(shown.unsqueeze(0)).softmax(1)[0]
        flipped = int(pc.argmax()) != int(pt.argmax())
        col.metric(p.stem, f"{int(pc.argmax())} → {int(pt.argmax())}")
        col.caption(f"{'🔴 prediction flipped' if flipped else '🟢 unchanged'} · "
                    f"{pt.max():.0%} confident with patch")

# ------------------------------------------------------------------ all candidates
with t_all:
    c1, c2 = st.columns([1, 3])
    only = c1.toggle("Suspicious only", value=bad)
    cls = c2.multiselect("Dominant class", list(range(NUM_CLASSES)))
    view = df[df["suspicious"]] if only else df
    if cls:
        view = view[view["dominant_class"].isin(cls)]
    st.dataframe(view, hide_index=True, width="stretch", height=420, column_config={
        "attack_success_rate": st.column_config.ProgressColumn("ASR", format="percent", min_value=0, max_value=1),
        "flip_rate": st.column_config.ProgressColumn("Flip rate", format="percent", min_value=0, max_value=1),
        "dominant_rate": st.column_config.NumberColumn("Dominant share", format="percent"),
        "mean_confidence": st.column_config.NumberColumn("Confidence", format="percent"),
    })
    st.caption(f"{len(view)} of {len(df)} candidates · pred_class_* = how many probe images landed in each class")

# ------------------------------------------------------------------ report
with t_rep:
    st.markdown("Download the results of **this scan**.")
    if st.button("Build HTML report"):
        with st.spinner("Rendering report..."):
            html_doc = build_html_report(model_path.name, model_path, model, images, df, s, asr_thr, flip_thr)
            out = REPORTS_DIR / f"scan_report_{model_path.stem}.html"
            out.write_text(html_doc, encoding="utf-8")
            st.session_state["report"] = (current_key, html_doc, str(out))
    rep = st.session_state.get("report")
    c1, c2, c3 = st.columns(3)
    if rep and rep[0] == current_key:
        c1.download_button("⬇ HTML report", rep[1], f"scan_report_{model_path.stem}.html", "text/html",
                           type="primary", width="stretch")
        st.caption(f"Saved to {rep[2]}. Open it in a browser and use Print → Save as PDF for a PDF.")
    else:
        c1.button("⬇ HTML report", disabled=True, width="stretch", help="Build the report first")
    c2.download_button("⬇ Candidates CSV", df.to_csv(index=False), f"scan_results_{model_path.stem}.csv",
                       "text/csv", width="stretch")
    c3.download_button("⬇ Summary JSON", summary_json(model_path.name, model_path, s, asr_thr, flip_thr),
                       f"scan_summary_{model_path.stem}.json", "application/json", width="stretch")
