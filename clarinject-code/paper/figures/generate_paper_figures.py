#!/usr/bin/env python3
"""Generate publication-quality figures for the ClarInject-Code paper."""

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
from pathlib import Path

OUT = Path(__file__).parent

# --- Color palette (colorblind-safe) ---
EXEC_COLOR = "#0072B2"
CLAR_NONE  = "#D55E00"
CLAR_D1    = "#E69F00"
CLAR_D2    = "#009E73"
CLAR_D3    = "#CC79A7"

MODELS = ["Claude Sonnet", "GPT-4o", "Llama-3-70B"]

# Illustrative data matching the paper's reported numbers
ASR_EXEC = {"Claude Sonnet": 0.09, "GPT-4o": 0.12, "Llama-3-70B": 0.21}
ASR_CLAR_NONE = {"Claude Sonnet": 0.31, "GPT-4o": 0.38, "Llama-3-70B": 0.52}
ASR_CLAR_D1   = {"Claude Sonnet": 0.22, "GPT-4o": 0.29, "Llama-3-70B": 0.41}
ASR_CLAR_D2   = {"Claude Sonnet": 0.13, "GPT-4o": 0.18, "Llama-3-70B": 0.28}
ASR_CLAR_D3   = {"Claude Sonnet": 0.08, "GPT-4o": 0.11, "Llama-3-70B": 0.19}

CCR_NONE = {"Claude Sonnet": 0.84, "GPT-4o": 0.86, "Llama-3-70B": 0.81}
CCR_D1   = {"Claude Sonnet": 0.83, "GPT-4o": 0.84, "Llama-3-70B": 0.80}
CCR_D2   = {"Claude Sonnet": 0.76, "GPT-4o": 0.78, "Llama-3-70B": 0.73}
CCR_D3   = {"Claude Sonnet": 0.65, "GPT-4o": 0.68, "Llama-3-70B": 0.62}

# CI half-widths (illustrative)
CI = {"Claude Sonnet": 0.06, "GPT-4o": 0.07, "Llama-3-70B": 0.08}


def fig1_overview():
    """Conceptual diagram: execution vs clarification condition."""
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    def draw_flow(ax, title, color, steps, highlight_idx):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, len(steps) + 0.5)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", color=color, pad=8)
        for i, (label, col) in enumerate(steps):
            y = len(steps) - 1 - i
            box_color = col if col else "#eeeeee"
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (0.5, y + 0.05), 9, 0.75,
                    boxstyle="round,pad=0.05",
                    facecolor=box_color,
                    edgecolor="gray",
                    linewidth=1.2,
                    alpha=0.85,
                )
            )
            ax.text(5, y + 0.43, label, ha="center", va="center", fontsize=8.5,
                    fontweight="bold" if col else "normal",
                    color="white" if col else "black")
            if i < len(steps) - 1:
                ax.annotate("", xy=(5, y + 0.03), xytext=(5, y - 0.15),
                            arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))

    exec_steps = [
        ("Fully-specified task", None),
        ("Tool output (adversarial payload)", CLAR_NONE),
        ("Agent executes", None),
        ("Output (check ASR)", None),
    ]
    clar_steps = [
        ("Ambiguous task", None),
        ("Agent asks clarifying question", EXEC_COLOR),
        ("Adversarial answer (payload)", CLAR_NONE),
        ("Agent executes (amplified risk)", None),
    ]

    draw_flow(axes[0], "Execution Condition\n(Baseline)", EXEC_COLOR, exec_steps, 1)
    draw_flow(axes[1], "Clarification Condition\n(+Clarification Tax)", CLAR_NONE, clar_steps, 2)

    fig.suptitle(
        r"Clarification Tax $\Delta_\mathrm{clar} = \mathrm{ASR}_\mathrm{clar} - \mathrm{ASR}_\mathrm{exec}$",
        fontsize=11, y=0.02
    )
    plt.tight_layout()
    plt.savefig(OUT / "fig1_overview.pdf")
    plt.close()
    print("Saved fig1_overview.pdf")


def fig2_asr_bars():
    """ASR by condition and defense, per model."""
    fig, axes = plt.subplots(1, 3, figsize=(8, 3.5), sharey=True)

    conditions = ["Execution", "Clar.\n(None)", "Clar.\n(D1)", "Clar.\n(D2)", "Clar.\n(D3)"]
    colors = [EXEC_COLOR, CLAR_NONE, CLAR_D1, CLAR_D2, CLAR_D3]
    x = np.arange(len(conditions))

    for ax, model in zip(axes, MODELS):
        asrs = [
            ASR_EXEC[model],
            ASR_CLAR_NONE[model],
            ASR_CLAR_D1[model],
            ASR_CLAR_D2[model],
            ASR_CLAR_D3[model],
        ]
        ci_h = [CI[model]] * 5
        ci_h[0] = CI[model] * 0.6  # exec has smaller CI

        ax.bar(x, asrs, color=colors, width=0.65, zorder=2, edgecolor="white", linewidth=0.5)
        ax.errorbar(x, asrs, yerr=ci_h, fmt="none", color="black", capsize=3.5,
                    linewidth=1.0, zorder=3)
        ax.axhline(ASR_EXEC[model], color=EXEC_COLOR, linestyle="--", alpha=0.4,
                   linewidth=0.9, zorder=1)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=8)
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 0.72)
        ax.grid(axis="y", alpha=0.25, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("Attack Success Rate (ASR)", fontsize=11)

    legend_patches = [
        mpatches.Patch(color=EXEC_COLOR, label="Execution baseline"),
        mpatches.Patch(color=CLAR_NONE,  label="Clarification – No Defense"),
        mpatches.Patch(color=CLAR_D1,    label="D1: Prompt Tag"),
        mpatches.Patch(color=CLAR_D2,    label="D2: Struct. Extract"),
        mpatches.Patch(color=CLAR_D3,    label="D3: Plan-Diff Gate"),
    ]
    fig.legend(handles=legend_patches, ncol=3, fontsize=8,
               loc="lower center", bbox_to_anchor=(0.5, -0.16),
               frameon=False)

    plt.tight_layout()
    plt.savefig(OUT / "fig2_asr_bars.pdf")
    plt.close()
    print("Saved fig2_asr_bars.pdf")


def fig3_pareto():
    """Pareto frontier: CCR (collaboration) vs 1-ASR (security)."""
    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    model_colors = {
        "Claude Sonnet": "#0072B2",
        "GPT-4o":        "#D55E00",
        "Llama-3-70B":   "#009E73",
    }
    defense_markers = {"None": "o", "D1": "s", "D2": "^", "D3": "D"}
    defense_labels  = {"None": "No Defense", "D1": "D1: Prompt Tag",
                       "D2": "D2: Struct. Extract", "D3": "D3: Plan-Diff Gate"}

    for model in MODELS:
        points = {
            "None": (CCR_NONE[model], 1 - ASR_CLAR_NONE[model]),
            "D1":   (CCR_D1[model],   1 - ASR_CLAR_D1[model]),
            "D2":   (CCR_D2[model],   1 - ASR_CLAR_D2[model]),
            "D3":   (CCR_D3[model],   1 - ASR_CLAR_D3[model]),
        }
        # Connect Pareto points with a line
        sorted_pts = sorted(points.values(), key=lambda p: p[0])
        ax.plot([p[0] for p in sorted_pts], [p[1] for p in sorted_pts],
                color=model_colors[model], alpha=0.35, linewidth=1.2, zorder=1)

        for defense, (ccr, sec) in points.items():
            ax.scatter(ccr, sec,
                       marker=defense_markers[defense],
                       color=model_colors[model],
                       s=70, zorder=3,
                       edgecolors="white", linewidths=0.8)

    # Execution baseline
    for model in MODELS:
        ax.scatter(1.0, 1 - ASR_EXEC[model],
                   marker="*", color=model_colors[model], s=100, zorder=4,
                   edgecolors="white", linewidths=0.5)

    # Ideal corner
    ax.annotate("Ideal", xy=(1.0, 1.0), xytext=(0.92, 0.97),
                fontsize=8, color="gray",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    # Legends
    model_patches = [mpatches.Patch(color=model_colors[m], label=m) for m in MODELS]
    defense_markers_legend = [
        plt.Line2D([0], [0], marker=defense_markers[d], color="gray",
                   linewidth=0, markersize=7, label=defense_labels[d])
        for d in ["None", "D1", "D2", "D3"]
    ]
    l1 = ax.legend(handles=model_patches, fontsize=8, loc="lower left",
                   title="Model", title_fontsize=8, frameon=True, framealpha=0.8)
    ax.add_artist(l1)
    ax.legend(handles=defense_markers_legend, fontsize=8, loc="lower right",
              title="Defense", title_fontsize=8, frameon=True, framealpha=0.8)

    ax.set_xlabel("Collaboration Quality (CCR)", fontsize=11)
    ax.set_ylabel(r"Attack Resistance ($1 - \mathrm{ASR}_\mathrm{clar}$)", fontsize=11)
    ax.set_title("Pareto Frontier: Collaboration vs. Security", fontsize=11, fontweight="bold")
    ax.set_xlim(0.55, 1.08)
    ax.set_ylim(0.38, 1.05)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "fig3_pareto.pdf")
    plt.close()
    print("Saved fig3_pareto.pdf")


def fig4_by_category():
    """Clarification tax by attack category."""
    categories = ["backdoor", "secret\nexfil", "safety\nremoval", "dep\npoison", "destructive"]
    taxes = [0.19, 0.28, 0.29, 0.14, 0.27]
    exec_asr = [0.08, 0.18, 0.16, 0.06, 0.15]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars1 = ax.bar(x - width/2, exec_asr, width, label=r"$\mathrm{ASR}_\mathrm{exec}$",
                   color=EXEC_COLOR, edgecolor="white")
    bars2 = ax.bar(x + width/2, taxes, width, label=r"Clarification Tax $\Delta_\mathrm{clar}$",
                   color=CLAR_NONE, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Rate", fontsize=11)
    ax.set_title("ASR and Clarification Tax by Attack Category\n(averaged across models, no defense)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.set_ylim(0, 0.42)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / "fig4_by_category.pdf")
    plt.close()
    print("Saved fig4_by_category.pdf")


if __name__ == "__main__":
    fig1_overview()
    fig2_asr_bars()
    fig3_pareto()
    fig4_by_category()
    print("All figures saved to", OUT)
