"""Generate IEEE paper figures as vector PDFs (no baked-in titles).

Style contract: single-column 3.5 in / double-column 7.16 in widths, all
fonts >= 8 pt, Okabe-Ito colorblind-safe palette, hatching + distinct
markers so every series survives grayscale, axis labels only (captions
live in main.tex), and label boxes placed on a fixed grid with margins
(no overlapping annotations).

    python scripts/v4_paper_figures.py
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJECT = Path(__file__).resolve().parent.parent
FIGURES = PROJECT / "paper" / "figures"
POSTHOC = PROJECT / "results" / "v4" / "posthoc"

# Okabe-Ito colorblind-safe palette.
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "pink": "#CC79A7", "sky": "#56B4E9", "brick": "#D55E00",
      "yellow": "#F0E442", "gray": "#999999"}

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,  # editable text in vector output
})


def save(name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / name
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"wrote {path.relative_to(PROJECT)}")


def fig_latch_timeline() -> None:
    """First reliability-entry times per Part B family (strip plot)."""
    frame = pd.read_csv(POSTHOC / "partB_run_classification.csv")
    entered = frame[frame["reliability_entries"] > 0].copy()
    families = ["bias", "dropout", "drift", "combined_bias_load", "load"]
    labels = ["bias", "dropout", "drift", "combined", "load"]
    onsets = {"bias": 2.0, "dropout": 2.0, "drift": 2.0,
              "combined_bias_load": 3.0, "load": 3.0}
    rng = __import__("numpy").random.default_rng(11)
    _, ax = plt.subplots(figsize=(3.5, 2.6))
    for row, family in enumerate(families):
        sub = entered[entered["family"] == family]
        jitter = rng.uniform(-0.22, 0.22, len(sub))
        pre = sub["label"] == "PRE_EVENT_LATCH"
        ax.scatter(sub[pre]["first_entry_time_s"], row + jitter[pre.to_numpy()],
                   s=14, marker="x", color=OI["brick"], linewidths=0.9,
                   label="pre-event latch" if row == 0 else None, zorder=3)
        ax.scatter(sub[~pre]["first_entry_time_s"],
                   row + jitter[(~pre).to_numpy()],
                   s=14, marker="o", facecolors="none",
                   edgecolors=OI["blue"], linewidths=0.9,
                   label="post-event entry" if row == 0 else None, zorder=3)
        ax.axvline(onsets[family], color=OI["gray"], linestyle="--",
                   linewidth=0.8, ymin=row / 5.0, ymax=(row + 0.9) / 5.0)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("first reliability-entry time (s)")
    ax.set_ylabel("fault family")
    ax.set_xlim(0, 6)
    ax.text(1.97, 4.62, "onset 2.0 s", fontsize=8, color=OI["gray"],
            ha="right", va="center")
    ax.text(3.03, 4.62, "onset 3.0 s", fontsize=8, color=OI["gray"],
            ha="left", va="center")
    ax.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    save("fig_latch_timeline.pdf")


def fig_conditional_detection() -> None:
    """Pooled (preregistered) vs conditional-on-clean detection by family."""
    relabeled = pd.read_csv(POSTHOC / "envelope_relabelled_posthoc.csv")
    families = ["bias", "dropout", "drift", "combined_bias_load"]
    labels = ["bias", "dropout", "drift", "combined"]
    pooled, conditional = [], []
    for family in families:
        sub = relabeled[relabeled["family"] == family]
        pooled.append(sub["preregistered_pooled_detection"].mean())
        conditional.append(sub["posthoc_cond_pooled_detection"].mean())
    x = __import__("numpy").arange(len(families))
    _, ax = plt.subplots(figsize=(3.5, 2.4))
    ax.bar(x - 0.2, pooled, width=0.4, color=OI["blue"], edgecolor="black",
           linewidth=0.7, label="pooled (preregistered)")
    ax.bar(x + 0.2, conditional, width=0.4, color=OI["orange"],
           edgecolor="black", linewidth=0.7, hatch="///",
           label="conditional on clean start (post-hoc)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("detection probability")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc="upper center",
              bbox_to_anchor=(0.5, 1.32), ncol=2)
    plt.tight_layout()
    save("fig_conditional_detection.pdf")


def fig_architecture() -> None:
    """Double-column block diagram of the V4 fault-tolerant loop.

    Layout was iterated against rendered previews: every label sits in
    whitespace with a white halo box, and no two text boxes overlap.
    Residual symbols ($e_m$, $e_w$) are defined in the paper caption.
    """
    _, ax = plt.subplots(figsize=(7.16, 3.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    def box(x, y, w, h, text):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                                    facecolor="white", edgecolor="black",
                                    linewidth=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=8, linespacing=1.3)

    def seg(x0, y0, x1, y1, head=True):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                     arrowstyle="-|>" if head else "-",
                                     mutation_scale=10, linewidth=0.9,
                                     color="black"))

    def tag(x, y, text, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.8})

    # Top row: reference -> MPC -> plant -> sensors.
    box(1, 34, 11, 10, "reference\nr")
    box(17, 34, 15, 10, "LSTM-MPC")
    box(37, 34, 15, 10, "DC motor\nplant")
    box(57, 34, 16, 10, "sensors:\nspeed, current")
    # Bottom row: estimators + monitor.
    box(14, 8, 14, 12, "main LSTM\nsensor")
    box(33, 8, 14, 12, "witness\naux LSTM/EKF")
    box(52, 8, 14, 12, "V4 monitor\nCUSUM +\nwitness AND")
    box(79, 19, 18, 10, "feedback\nselector")
    # Top-row signals.
    seg(12, 39, 17, 39)
    tag(14.5, 40.6, "r")
    seg(32, 39, 37, 39)
    tag(34.5, 40.6, "u")
    seg(52, 39, 57, 39)
    tag(54.5, 40.6, "y, i")
    # Sensors -> main estimator bus.
    seg(62, 34, 62, 24, head=False)
    seg(62, 24, 25, 24, head=False)
    seg(25, 24, 25, 20)
    tag(43.5, 25.6, "y, V, i")
    # Sensors -> witness bus.
    seg(68, 34, 68, 22, head=False)
    seg(68, 22, 40, 22, head=False)
    seg(40, 22, 40, 20)
    ax.text(66.8, 28, "V, i", ha="right", va="center", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    # Residuals into the monitor.
    seg(28, 14, 33, 14)
    tag(30.5, 15.4, "$e_m$")
    seg(47, 14, 52, 14)
    tag(49.5, 15.4, "$e_w$")
    # Monitor -> selector verdict.
    seg(66, 14, 72, 14, head=False)
    seg(72, 14, 72, 24, head=False)
    seg(72, 24, 79, 24)
    ax.text(74, 16, "suspect", ha="left", va="center", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    # Main estimate up to MPC.
    seg(21, 20, 21, 34)
    ax.text(19.8, 27, "y hat", ha="right", va="center", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
    # Healthy measurement to the selector.
    seg(73, 39, 88, 39, head=False)
    seg(88, 39, 88, 29)
    tag(80.5, 40.4, "y")
    # Estimate along the bottom bus to the selector.
    seg(21, 8, 21, 3, head=False)
    seg(21, 3, 93, 3, head=False)
    seg(93, 3, 93, 19)
    tag(57, 4.4, "y hat")
    # Selector verdict + feedback to MPC.
    seg(92, 29, 92, 47, head=False)
    seg(92, 47, 24.5, 47, head=False)
    seg(24.5, 47, 24.5, 44)
    tag(58, 48.2, "feedback")
    save("fig_architecture.pdf")


def main() -> None:
    fig_latch_timeline()
    fig_conditional_detection()
    fig_architecture()


if __name__ == "__main__":
    main()
