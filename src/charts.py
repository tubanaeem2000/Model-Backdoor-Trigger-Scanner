"""
charts.py - Interactive (hover-enabled) Altair charts for the Streamlit dashboard.
Static matplotlib versions for the downloadable report live in visualize.py.
"""
import altair as alt
import pandas as pd

from src.config import NUM_CLASSES

# Theme
TEXT, MUTED, GRID, PANEL = "#E6FFFA", "#7A9490", "#16302C", "#0F1E1C"
# Categorical series colours (validated for the dark background: lightness, chroma, colour-blind safety)
SERIES = ["#0D9488", "#D97706", "#8B7CF0", "#D55181"]
THRESHOLD_COLOR = "#F87171"


def _style(chart, height=280):
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                            domainColor=GRID, tickColor=GRID, labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=TEXT, titleColor=MUTED, orient="top")
            .configure_title(color=TEXT, anchor="start", fontSize=13))


def _colors(names):
    return alt.Scale(domain=list(names), range=SERIES[:len(names)])


def before_after_distribution(base_preds, new_preds):
    """Grouped bars: how many probe images fall in each class, before vs after the trigger."""
    rows = []
    for stage, preds in [("Before trigger", base_preds), ("After trigger", new_preds)]:
        counts = pd.Series(preds.tolist()).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        rows += [{"stage": stage, "class": c, "images": int(n)} for c, n in counts.items()]
    df = pd.DataFrame(rows)
    stages = ["Before trigger", "After trigger"]
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=3, width={"band": 0.4}).encode(
        x=alt.X("class:O", title="Predicted class", axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("stage:N", sort=stages),
        y=alt.Y("images:Q", title="Images"),
        color=alt.Color("stage:N", scale=_colors(stages), sort=stages, title=None),
        tooltip=["stage", "class", "images"],
    ).properties(title="Prediction distribution before vs after the trigger")
    return _style(chart)


def asr_heatmap(df):
    """Colour x position grid; cell = best ASR over all patch sizes. Hover for details."""
    best = (df.sort_values("attack_success_rate", ascending=False)
              .groupby(["color", "position"], as_index=False).first())
    base = alt.Chart(best).encode(
        x=alt.X("position:N", title=None, axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("color:N", title=None, axis=alt.Axis(labelOverlap=False)))
    rect = base.mark_rect(cornerRadius=3, stroke="#070B0B", strokeWidth=2).encode(
        color=alt.Color("attack_success_rate:Q", title="ASR",
                        scale=alt.Scale(domain=[0, 1], range=[PANEL, "#0D5C55", "#14B8A6", "#5EEAD4"])),
        tooltip=["candidate_id", "dominant_class",
                 alt.Tooltip("attack_success_rate:Q", format=".1%", title="ASR"),
                 alt.Tooltip("flip_rate:Q", format=".1%", title="Flip rate"), "suspicious"])
    text = base.mark_text(fontSize=11).encode(
        text=alt.Text("attack_success_rate:Q", format=".0%"),
        color=alt.condition("datum.attack_success_rate > 0.6", alt.value("#070B0B"), alt.value(TEXT)))
    return _style((rect + text).properties(title="Best attack success rate per colour × position"), 400)


def sorted_asr_curves(scans, threshold):
    """One line per model: ASR of all candidates, sorted high -> low. Backdoors show a block above the line."""
    rows = []
    for name, df in scans.items():
        d = df.sort_values("attack_success_rate", ascending=False).reset_index(drop=True)
        rows += [{"model": name, "rank": i + 1, "ASR": r.attack_success_rate,
                  "candidate": r.candidate_id, "target": r.dominant_class} for i, r in d.iterrows()]
    data = pd.DataFrame(rows)
    names = list(scans)
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(
        x=alt.X("rank:Q", title="Candidate rank (sorted by ASR)"),
        y=alt.Y("ASR:Q", title="Attack success rate", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=_colors(names), title=None))
    hover = alt.selection_point(fields=["rank"], nearest=True, on="pointerover", empty=False)
    points = alt.Chart(data).mark_circle(size=64).encode(
        x="rank:Q", y="ASR:Q", color=alt.Color("model:N", scale=_colors(names), title=None),
        opacity=alt.condition(hover, alt.value(1), alt.value(0)),
        tooltip=["model", "rank", "candidate", "target", alt.Tooltip("ASR:Q", format=".1%")],
    ).add_params(hover)
    rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=THRESHOLD_COLOR, strokeDash=[5, 4]).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y": [threshold], "x": [len(data) / max(len(names), 1)],
                                    "t": [f"suspicious ≥ {threshold:.0%}"]})).mark_text(
        align="right", dy=-7, color=THRESHOLD_COLOR, fontSize=11).encode(x="x:Q", y="y:Q", text="t:N")
    return _style((lines + points + rule + label).properties(title="ASR of every candidate trigger"))


def position_comparison(scans, threshold):
    """Grouped bars: best ASR per patch position, one bar per model."""
    rows = []
    for name, df in scans.items():
        g = df.groupby("position")["attack_success_rate"].max()
        rows += [{"model": name, "position": p, "best ASR": v} for p, v in g.items()]
    data = pd.DataFrame(rows)
    names = list(scans)
    bars = alt.Chart(data).mark_bar(cornerRadiusEnd=3, width={"band": 0.8 / len(names)}).encode(
        x=alt.X("position:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=names),
        y=alt.Y("best ASR:Q", title="Best attack success rate", axis=alt.Axis(format="%"),
                scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=_colors(names), sort=names, title=None),
        tooltip=["model", "position", alt.Tooltip("best ASR:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"y": [threshold]})).mark_rule(
        color=THRESHOLD_COLOR, strokeDash=[5, 4]).encode(y="y:Q")
    return _style((bars + rule).properties(title="Where does a patch hijack the model?"))


def training_comparison(results):
    """Grouped bars from reports/training_results.json: clean accuracy vs attack success rate."""
    rows = []
    for model in ("clean", "backdoored"):
        rows.append({"model": f"{model}_model", "metric": "Clean accuracy",
                     "value": results[model]["clean_accuracy"] / 100})
        rows.append({"model": f"{model}_model", "metric": "Attack success rate",
                     "value": results[model]["attack_success_rate"] / 100})
    data = pd.DataFrame(rows)
    names = ["backdoored_model", "clean_model"]
    chart = alt.Chart(data).mark_bar(cornerRadiusEnd=3, width={"band": 0.4}).encode(
        x=alt.X("metric:N", title=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model:N", sort=names),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=_colors(names), sort=names, title=None),
        tooltip=["model", "metric", alt.Tooltip("value:Q", format=".2%")],
    ).properties(title="Training results: same accuracy, very different behaviour")
    return _style(chart)
