"""
visualize.py - All plots, in the project's teal-green + black theme.
"""
import matplotlib
matplotlib.use("Agg")  # draw without opening a window (works everywhere)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------- Theme ----------------
BG = "#070B0B"        # near-black background
PANEL = "#0F1E1C"     # dark teal-black panel
TEAL = "#14B8A6"      # main accent
TEAL_LIGHT = "#5EEAD4"
TEXT = "#E6FFFA"      # primary text
MUTED = "#7A9490"     # secondary text / axes
BAR_MUTED = "#2F4744" # non-highlighted bars
RED = "#F87171"       # used only for "wrong / suspicious" text

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": MUTED, "axes.titlecolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": BAR_MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
})
# Sequential colormap: one hue (teal), dark = low, bright = high
TEAL_CMAP = LinearSegmentedColormap.from_list("teal_seq", [PANEL, "#0D5C55", TEAL, TEAL_LIGHT])


def _show(ax, image):
    """Display a [C, 28, 28] tensor (1 or 3 channels)."""
    if image.shape[0] == 1:
        ax.imshow(image.squeeze(), cmap="gray")
    else:
        ax.imshow(image.permute(1, 2, 0).clamp(0, 1))  # [3,28,28] -> [28,28,3]


def plot_samples(dataset, n=16, cols=8, indices=None, title=None):
    """Return a Figure showing n images of the dataset (or the given indices) with labels."""
    indices = list(indices) if indices is not None else list(range(n))
    n = len(indices)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.5), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            image, label = dataset[indices[i]]
            _show(ax, image)
            ax.set_title(f"Label: {label}", fontsize=9)
    if title:
        fig.suptitle(title, color=TEAL_LIGHT)
    fig.tight_layout()
    return fig


# ---------------- Scanner plots ----------------
def plot_evidence(clean_images, patched_images, base_preds, new_preds, title, n=8):
    """Top row: original image + prediction. Bottom row: with candidate trigger + new prediction."""
    n = min(n, len(clean_images))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.3, 3.3), squeeze=False)
    for i in range(n):
        for r, (imgs, preds) in enumerate([(clean_images, base_preds), (patched_images, new_preds)]):
            ax = axes[r][i]
            ax.axis("off")
            _show(ax, imgs[i])
            changed = r == 1 and int(preds[i]) != int(base_preds[i])
            ax.set_title(f"pred {int(preds[i])}", fontsize=9, color=RED if changed else TEXT)
    axes[0][0].text(-8, 14, "Original", rotation=90, va="center", color=MUTED)
    axes[1][0].text(-8, 14, "+ Trigger", rotation=90, va="center", color=MUTED)
    fig.suptitle(title, color=TEAL_LIGHT)
    fig.tight_layout()
    return fig


def plot_class_distribution(row, num_classes=10):
    """Bar chart: how the patched images were classified. Dominant class in teal."""
    counts = [row[f"pred_class_{c}"] for c in range(num_classes)]
    dom = int(row["dominant_class"])
    fig, ax = plt.subplots(figsize=(6, 2.8))
    colors = [TEAL if c == dom else BAR_MUTED for c in range(num_classes)]
    ax.bar(range(num_classes), counts, color=colors, width=0.6)
    ax.text(dom, counts[dom], f"{counts[dom]} ({counts[dom] / sum(counts):.0%})",
            ha="center", va="bottom", color=TEXT)
    ax.set_ylim(0, max(counts) * 1.15)
    ax.set_xticks(range(num_classes))
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Images")
    ax.set_title(f"Predictions with trigger: {row['candidate_id']}", loc="left")
    ax.yaxis.grid(True, color=PANEL)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_asr_heatmap(df):
    """Heatmap: best attack success rate (over all sizes) for each colour x position."""
    table = df.pivot_table(index="color", columns="position", values="attack_success_rate", aggfunc="max")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    im = ax.imshow(table.values, cmap=TEAL_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns, rotation=20)
    ax.set_yticks(range(len(table.index)), table.index)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color=BG if v > 0.6 else TEXT)
    for s in ax.spines.values():
        s.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    cbar.outline.set_visible(False)
    ax.set_title("Max attack success rate by colour x position", loc="left")
    fig.tight_layout()
    return fig


def plot_trigger(cand, example_image, patched_example):
    """Three panels: the suspected trigger alone, an example input, and the input + trigger."""
    import torch
    from src.scanner import apply_candidate
    canvas = apply_candidate(torch.zeros(1, 3, 28, 28), cand)[0]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.6))
    for ax, img, title in zip(axes, [canvas, example_image, patched_example],
                              ["Suspected trigger", "Clean input", "Input + trigger"]):
        _show(ax, img)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(BAR_MUTED)
        ax.set_title(title, fontsize=9)
    fig.suptitle(cand["candidate_id"], color=TEAL_LIGHT)
    fig.tight_layout()
    return fig
