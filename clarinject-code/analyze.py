#!/usr/bin/env python3
"""
analyze.py — Generate figures and tables from ClarInject-Code results.

Usage:
    python analyze.py --input results/metrics.json --output results/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

COLORBLIND_PALETTE = {
    "execution": "#0072B2",
    "clarification_none": "#D55E00",
    "clarification_prompt_tag": "#E69F00",
    "clarification_structured_extract": "#009E73",
    "clarification_plan_diff_gate": "#CC79A7",
}

DEFENSE_LABELS = {
    "none": "No Defense",
    "prompt_tag": "Prompt Tag",
    "structured_extract": "Struct. Extract",
    "plan_diff_gate": "Plan-Diff Gate",
}


def load_metrics(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def figure_asr_by_condition(metrics: dict, out_dir: Path) -> None:
    """Bar chart: ASR by condition (execution vs clarification), per model."""
    agg = metrics["aggregated"]
    models = sorted({r["model"] for r in agg})
    defenses = sorted({r["defense"] for r in agg})

    fig, axes = plt.subplots(
        1, len(models), figsize=(4 * len(models), 4), sharey=True
    )
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        exec_row = next(
            (r for r in agg if r["model"] == model and r["condition"] == "execution" and r["defense"] == "none"),
            None,
        )
        exec_asr = exec_row["asr"] if exec_row else 0.0
        exec_lo = exec_row["asr_ci_lo"] if exec_row else 0.0
        exec_hi = exec_row["asr_ci_hi"] if exec_row else 0.0

        x_labels = ["Execution\n(baseline)"]
        asrs = [exec_asr]
        errs_lo = [exec_asr - exec_lo]
        errs_hi = [exec_hi - exec_asr]
        colors = [COLORBLIND_PALETTE["execution"]]

        for defense in defenses:
            clar_row = next(
                (
                    r for r in agg
                    if r["model"] == model
                    and r["condition"] == "clarification"
                    and r["defense"] == defense
                ),
                None,
            )
            if clar_row:
                x_labels.append(f"Clarification\n({DEFENSE_LABELS.get(defense, defense)})")
                asrs.append(clar_row["asr"])
                errs_lo.append(clar_row["asr"] - clar_row["asr_ci_lo"])
                errs_hi.append(clar_row["asr_ci_hi"] - clar_row["asr"])
                key = f"clarification_{defense}"
                colors.append(COLORBLIND_PALETTE.get(key, "#999999"))

        x = np.arange(len(x_labels))
        bars = ax.bar(x, asrs, color=colors, width=0.6, zorder=2)
        ax.errorbar(
            x,
            asrs,
            yerr=[errs_lo, errs_hi],
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1.2,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(exec_asr, color=COLORBLIND_PALETTE["execution"], linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.3, zorder=0)

    axes[0].set_ylabel("Attack Success Rate (ASR)", fontsize=10)
    fig.suptitle("ASR by Condition and Defense", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "fig1_asr_by_condition.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def figure_pareto_frontier(metrics: dict, out_dir: Path) -> None:
    """Pareto frontier: collaboration quality (CCR) vs attack resistance (1-ASR)."""
    agg = metrics["aggregated"]
    models = sorted({r["model"] for r in agg})
    defenses = sorted({r["defense"] for r in agg})

    fig, ax = plt.subplots(figsize=(6, 4.5))

    markers = {"none": "o", "prompt_tag": "s", "structured_extract": "^", "plan_diff_gate": "D"}
    color_cycle = ["#D55E00", "#E69F00", "#009E73", "#CC79A7"]

    for model_idx, model in enumerate(models):
        xs, ys, labels = [], [], []
        for defense in defenses:
            row = next(
                (
                    r for r in agg
                    if r["model"] == model
                    and r["condition"] == "clarification"
                    and r["defense"] == defense
                ),
                None,
            )
            if row:
                xs.append(row.get("ccr", 1 - row["asr"]))
                ys.append(1 - row["asr"])
                labels.append(defense)

        for x, y, defense in zip(xs, ys, labels):
            ax.scatter(
                x, y,
                marker=markers.get(defense, "o"),
                color=color_cycle[model_idx % len(color_cycle)],
                s=80,
                zorder=3,
                label=f"{model} / {DEFENSE_LABELS.get(defense, defense)}" if model_idx == 0 else "",
            )
            ax.annotate(
                DEFENSE_LABELS.get(defense, defense),
                (x, y),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

    ax.set_xlabel("Collaboration Quality (CCR)", fontsize=11)
    ax.set_ylabel("Attack Resistance (1 − ASR)", fontsize=11)
    ax.set_title("Pareto Frontier: Collaboration vs. Security", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    # Legend patches for models
    patches = [
        mpatches.Patch(color=color_cycle[i], label=m)
        for i, m in enumerate(models)
    ]
    ax.legend(handles=patches, fontsize=8, loc="lower left")

    plt.tight_layout()
    out_path = out_dir / "fig2_pareto_frontier.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def figure_clarification_tax(metrics: dict, out_dir: Path) -> None:
    """Bar chart of clarification tax per model, grouped by defense."""
    taxes = metrics["clarification_tax"]
    if not taxes:
        print("No clarification tax data to plot.")
        return

    models = sorted({t["model"] for t in taxes})
    defenses = sorted({t["defense"] for t in taxes})

    x = np.arange(len(models))
    width = 0.8 / max(len(defenses), 1)
    colors = ["#D55E00", "#E69F00", "#009E73", "#CC79A7"]

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, defense in enumerate(defenses):
        tax_vals = []
        for model in models:
            row = next(
                (t for t in taxes if t["model"] == model and t["defense"] == defense),
                None,
            )
            tax_vals.append(row["clarification_tax"] if row else 0.0)

        offset = (i - len(defenses) / 2 + 0.5) * width
        ax.bar(
            x + offset,
            tax_vals,
            width=width * 0.9,
            color=colors[i % len(colors)],
            label=DEFENSE_LABELS.get(defense, defense),
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Clarification Tax (ASRₙₐᵣ − ASRₑₓₑₙ)", fontsize=10)
    ax.set_title("Clarification Tax by Model and Defense", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = out_dir / "fig3_clarification_tax.pdf"
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved {out_path}")


def print_markdown_table(metrics: dict) -> str:
    agg = metrics["aggregated"]
    header = "| Model | Condition | Defense | ASR | TSR | CCR | Turns |"
    sep = "|-------|-----------|---------|-----|-----|-----|-------|"
    rows = [header, sep]
    for r in sorted(agg, key=lambda x: (x["model"], x["condition"], x["defense"])):
        rows.append(
            f"| {r['model']} | {r['condition']} | {DEFENSE_LABELS.get(r['defense'], r['defense'])} "
            f"| {r['asr']:.3f} [{r['asr_ci_lo']:.3f}–{r['asr_ci_hi']:.3f}] "
            f"| {r['tsr']:.3f} | {r['ccr']:.3f} | {r['avg_turns']:.1f} |"
        )
    table = "\n".join(rows)
    print("\n=== RESULTS TABLE ===")
    print(table)
    return table


def main():
    parser = argparse.ArgumentParser(description="ClarInject-Code analysis and plotting")
    parser.add_argument("--input", default="results/metrics.json")
    parser.add_argument("--output", default="results/figures")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(args.input)

    figure_asr_by_condition(metrics, out_dir)
    figure_pareto_frontier(metrics, out_dir)
    figure_clarification_tax(metrics, out_dir)
    table = print_markdown_table(metrics)

    with open(out_dir / "results_table.md", "w") as f:
        f.write(table + "\n")
    print(f"\nResults table saved to {out_dir / 'results_table.md'}")


if __name__ == "__main__":
    main()
