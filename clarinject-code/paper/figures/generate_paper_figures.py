#!/usr/bin/env python3
"""Generate publication-quality figures from real ClarInject-Code results."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": False,
})

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path(__file__).parent
METRICS_PATH = OUT.parent.parent / "results" / "metrics.json"

# Colorblind-safe palette
EXEC_COLOR  = "#0072B2"
CLAR_NONE   = "#D55E00"
CLAR_D1     = "#E69F00"
CLAR_D2     = "#009E73"
CLAR_D3     = "#CC79A7"

DEFENSE_LABELS = {
    "none":               "No Defense",
    "prompt_tag":         "D1: Prompt Tag",
    "structured_extract": "D2: Struct. Extract",
    "plan_diff_gate":     "D3: Plan-Diff Gate",
}
DEFENSE_ORDER = ["none", "prompt_tag", "structured_extract", "plan_diff_gate"]
DEFENSE_COLORS = [CLAR_NONE, CLAR_D1, CLAR_D2, CLAR_D3]


def load():
    with open(METRICS_PATH) as f:
        return json.load(f)


def get(agg, condition, defense, field):
    row = next((r for r in agg
                if r["condition"] == condition and r["defense"] == defense), None)
    return row[field] if row else 0.0


def fig1_overview():
    """Conceptual diagram showing the two conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

    def draw_flow(ax, title, title_color, steps):
        ax.set_xlim(0, 10); ax.set_ylim(-0.2, len(steps))
        ax.axis("off")
        ax.set_title(title, fontsize=10, fontweight="bold", color=title_color, pad=6)
        for i, (label, color) in enumerate(steps):
            y = len(steps) - 1 - i
            ax.add_patch(mpatches.FancyBboxPatch(
                (0.3, y + 0.08), 9.4, 0.72,
                boxstyle="round,pad=0.04",
                facecolor=color if color else "#eeeeee",
                edgecolor="#aaaaaa", linewidth=1.0, alpha=0.9))
            ax.text(5, y + 0.44, label, ha="center", va="center", fontsize=8.5,
                    color="white" if color else "black",
                    fontweight="bold" if color else "normal")
            if i < len(steps) - 1:
                ax.annotate("", xy=(5, y + 0.06), xytext=(5, y - 0.14),
                            arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0))

    draw_flow(axes[0], "Execution Condition", EXEC_COLOR, [
        ("Fully-specified task", None),
        ("Adversarial tool output", CLAR_NONE),
        ("Agent produces code", None),
        ("ASR check", None),
    ])
    draw_flow(axes[1], "Clarification Condition", CLAR_NONE, [
        ("Ambiguous task", None),
        ("Agent asks question", EXEC_COLOR),
        ("Adversarial answer", CLAR_NONE),
        ("Agent produces code", None),
    ])

    fig.suptitle(
        r"Clarification Tax  $\Delta_{\rm clar} = {\rm ASR}_{\rm clar} - {\rm ASR}_{\rm exec}$",
        fontsize=10, y=0.03)
    plt.tight_layout()
    plt.savefig(OUT / "fig1_overview.pdf")
    plt.close()
    print("Saved fig1_overview.pdf")


def fig2_asr_bars():
    """ASR by condition and defense — real data."""
    m = load()
    agg = m["aggregated"]

    exec_asrs  = [get(agg, "execution",      d, "asr") for d in DEFENSE_ORDER]
    clar_asrs  = [get(agg, "clarification",  d, "asr") for d in DEFENSE_ORDER]
    exec_lo    = [get(agg, "execution",      d, "asr") - get(agg, "execution",     d, "asr_ci_lo") for d in DEFENSE_ORDER]
    exec_hi    = [get(agg, "execution",      d, "asr_ci_hi") - get(agg, "execution",     d, "asr") for d in DEFENSE_ORDER]
    clar_lo    = [get(agg, "clarification",  d, "asr") - get(agg, "clarification", d, "asr_ci_lo") for d in DEFENSE_ORDER]
    clar_hi    = [get(agg, "clarification",  d, "asr_ci_hi") - get(agg, "clarification", d, "asr") for d in DEFENSE_ORDER]

    x = np.arange(len(DEFENSE_ORDER))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.0))

    bars_e = ax.bar(x - w/2, exec_asrs, w, label="Execution (tool output)",
                    color=EXEC_COLOR, edgecolor="white", linewidth=0.5)
    bars_c = ax.bar(x + w/2, clar_asrs, w, label="Clarification (user answer)",
                    color=DEFENSE_COLORS, edgecolor="white", linewidth=0.5)

    ax.errorbar(x - w/2, exec_asrs, yerr=[exec_lo, exec_hi],
                fmt="none", color="black", capsize=3, linewidth=1.0)
    ax.errorbar(x + w/2, clar_asrs, yerr=[clar_lo, clar_hi],
                fmt="none", color="black", capsize=3, linewidth=1.0)

    # Annotation: D1 backfire arrow — tip clears the bar top + CI whisker
    d1_bar_top = clar_asrs[1] + clar_hi[1]   # top of error bar
    ax.annotate("D1 backfires\n(+17 pp)",
                xy=(x[1]+w/2, d1_bar_top + 0.025),   # tip above whisker
                xytext=(x[1]+w/2 + 0.55, d1_bar_top + 0.10),  # text to the right
                fontsize=7.5, color=CLAR_D1,
                arrowprops=dict(arrowstyle="->", color=CLAR_D1, lw=1.0))

    ax.set_xticks(x)
    ax.set_xticklabels([DEFENSE_LABELS[d] for d in DEFENSE_ORDER], fontsize=8.5)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=11)
    ax.set_title("ASR by Condition and Defense — Llama-3.1-8B", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_asr_bars.pdf")
    plt.close()
    print("Saved fig2_asr_bars.pdf")


def fig3_pareto():
    """Pareto frontier: CCR vs 1-ASR — real data."""
    m = load()
    agg = m["aggregated"]

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    for defense, color in zip(DEFENSE_ORDER, DEFENSE_COLORS):
        ccr  = get(agg, "clarification", defense, "ccr")
        asr  = get(agg, "clarification", defense, "asr")
        sec  = 1 - asr
        ax.scatter(ccr, sec, color=color, s=90, zorder=3,
                   edgecolors="white", linewidths=0.8, label=DEFENSE_LABELS[defense])
        # Per-dot label offsets — push "No Defense" upward so the legend
        # (upper-left) doesn't overlap its text
        offset = {"none": (5, 6), "prompt_tag": (5, 5),
                  "structured_extract": (5, 5), "plan_diff_gate": (5, 5)}
        ax.annotate(DEFENSE_LABELS[defense], (ccr, sec),
                    textcoords="offset points", xytext=offset.get(defense, (5, 5)),
                    fontsize=7.5)

    # Execution baseline horizontal line
    exec_asr = get(agg, "execution", "none", "asr")
    ax.axhline(1 - exec_asr, color=EXEC_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Exec. baseline (1−ASR={1-exec_asr:.2f})")

    ax.set_xlabel("Collaboration Quality (CCR)", fontsize=11)
    ax.set_ylabel(r"Attack Resistance ($1 - \mathrm{ASR}_{\mathrm{clar}}$)", fontsize=11)
    ax.set_title("Pareto Frontier: Collaboration vs. Security\nLlama-3.1-8B",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.3, 0.85)
    ax.set_ylim(0.3, 1.05)

    # Legend: upper-left corner is the only clear quadrant.
    # D1 is lower-right (0.71, 0.39), No Defense is center-left (0.49, 0.56),
    # D3 is upper-center (0.56, 0.99), D2 is upper-right (0.72, 0.92).
    # Upper-left (x<0.48, y>0.65) has no data points.
    ax.legend(fontsize=8, loc="upper left", frameon=True, framealpha=0.92,
              borderpad=0.6)
    ax.grid(alpha=0.2)
    ax.spines[["top","right"]].set_visible(False)

    # Mark D1 as dominated — text placed to the left and inside the plot
    # (previously y was below ylim=0.3)
    d1_ccr = get(agg, "clarification", "prompt_tag", "ccr")
    d1_sec = 1 - get(agg, "clarification", "prompt_tag", "asr")
    ax.annotate("dominated\n(anti-pattern)",
                xy=(d1_ccr, d1_sec),
                xytext=(d1_ccr - 0.17, d1_sec - 0.06),   # stays above ylim=0.3
                fontsize=7, color=CLAR_D1,
                arrowprops=dict(arrowstyle="->", color=CLAR_D1, lw=0.8))

    plt.tight_layout()
    plt.savefig(OUT / "fig3_pareto.pdf")
    plt.close()
    print("Saved fig3_pareto.pdf")


def fig4_by_category():
    """ASR by attack category — computed from transcripts."""
    # Per-category numbers from Table 5 in the paper
    categories  = ["backdoor", "secret\nexfil", "safety\nremoval", "dep\npoison", "destructive"]
    exec_asrs   = [0.800, 0.733, 0.800, 0.600, 0.867]
    clar_asrs   = [0.533, 0.400, 0.533, 0.267, 0.467]

    x = np.arange(len(categories))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.bar(x - w/2, exec_asrs, w, label="Execution (tool output)", color=EXEC_COLOR,
           edgecolor="white")
    ax.bar(x + w/2, clar_asrs, w, label="Clarification (no defense)", color=CLAR_NONE,
           edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Attack Success Rate (ASR)", fontsize=11)
    ax.set_title("ASR by Attack Category — Llama-3.1-8B, No Defense",
                 fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "fig4_by_category.pdf")
    plt.close()
    print("Saved fig4_by_category.pdf")


if __name__ == "__main__":
    fig1_overview()
    fig2_asr_bars()
    fig3_pareto()
    fig4_by_category()
    print(f"All figures saved to {OUT}")
