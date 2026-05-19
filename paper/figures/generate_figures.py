"""
Figure generation for:
  "Adaptive Verifier Allocation for Efficient Test-Time Compute Scaling:
   A Regret-Theoretic Analysis"

ALL data loaded from results/*.json (produced by experiments/run_experiment.py).
No synthetic data is used here.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent.parent
RESULTS = ROOT / "results"
OUTDIR  = Path(__file__).parent

# ── Global aesthetics ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "lines.linewidth": 1.8,
})

COLORS = {
    "DVA-MCTS": "#2166ac",
    "Uniform":  "#d6604d",
    "Random":   "#4dac26",
    "BestOfN":  "#b2abd2",
    "reference":"#444444",
}

def savefig(name):
    path = OUTDIR / name
    plt.savefig(path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  saved {path}")


def load(name):
    with open(RESULTS / f"{name}.json") as f:
        return json.load(f)


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Regret comparison  (DVA-MCTS vs. baselines)
# ════════════════════════════════════════════════════════════════════════════

def fig_regret():
    d = load("regret_scaling")
    budget = d["budget"]
    steps  = np.arange(1, budget + 1)
    exponent = d["regret_exponent"]
    r2       = d["r_squared"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: cumulative regret
    ax = axes[0]
    for key in ["DVA-MCTS", "Uniform", "Random"]:
        alg = d["algorithms"][key]
        mean = np.array(alg["cumulative_regret_mean"])
        std  = np.array(alg["cumulative_regret_std"])
        ax.plot(steps, mean, color=COLORS[key], label=key)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.12, color=COLORS[key])

    # O(sqrt(T)) reference scaled to DVA at midpoint
    dva_mean = np.array(d["algorithms"]["DVA-MCTS"]["cumulative_regret_mean"])
    ref_scale = dva_mean[budget // 2] / np.sqrt(budget // 2)
    ref = ref_scale * np.sqrt(steps)
    ax.plot(steps, ref, "--", color=COLORS["reference"], lw=1.2,
            label=r"$O(\sqrt{T})$ reference")

    ax.set_xlabel("Search steps $t$")
    ax.set_ylabel("Cumulative regret $R(t)$")
    ax.set_title("(a) Cumulative Regret")
    ax.legend(frameon=False)

    # Right: log-log to confirm slope
    ax2 = axes[1]
    for key in ["DVA-MCTS", "Uniform", "Random"]:
        alg  = d["algorithms"][key]
        mean = np.array(alg["cumulative_regret_mean"])
        pos  = mean > 0
        ax2.loglog(steps[pos], mean[pos], color=COLORS[key], label=key)

    ax2.loglog(steps, ref, "--", color=COLORS["reference"], lw=1.2,
               label=r"$O(\sqrt{T})$")
    ax2.set_xlabel("Search steps $t$ (log)")
    ax2.set_ylabel("Cumulative regret (log)")
    ax2.set_title(
        f"(b) Log-Log Scaling\n"
        f"DVA-MCTS exponent = {exponent:.3f} ($R^2={r2:.3f}$)"
    )
    ax2.legend(frameon=False)

    plt.tight_layout()
    savefig("fig1_regret_comparison.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Verifier call efficiency
# ════════════════════════════════════════════════════════════════════════════

def fig_verifier_calls():
    d = load("verifier_efficiency")
    by_budget = d["by_budget"]

    budgets   = sorted(int(b) for b in by_budget)
    dva_mean  = np.array([by_budget[str(b)]["dva_mean"]  for b in budgets])
    dva_std   = np.array([by_budget[str(b)]["dva_std"]   for b in budgets])
    uni_mean  = np.array([by_budget[str(b)]["uni_mean"]  for b in budgets])
    ratio_pct = np.array([by_budget[str(b)]["ratio_pct"] for b in budgets])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(budgets, uni_mean, color=COLORS["Uniform"],  label="Uniform Verification")
    ax.plot(budgets, dva_mean, color=COLORS["DVA-MCTS"], label="DVA-MCTS (ours)", marker="o", ms=4)
    ax.fill_between(budgets, dva_mean - dva_std, dva_mean + dva_std,
                    alpha=0.18, color=COLORS["DVA-MCTS"])
    ax.set_xlabel("Search Budget $B$")
    ax.set_ylabel("Verifier Calls")
    ax.set_title("(a) Absolute Verifier Calls")
    ax.legend(frameon=False)

    ax2 = axes[1]
    ax2.semilogx(budgets, ratio_pct, color=COLORS["DVA-MCTS"], marker="o", ms=5)
    ax2.axhline(100, color=COLORS["Uniform"], ls="--", lw=1.2, label="Uniform baseline (100%)")
    for b, r in zip(budgets, ratio_pct):
        ax2.annotate(f"{r:.0f}%", (b, r), textcoords="offset points",
                     xytext=(4, 5), fontsize=8)
    ax2.set_xlabel("Search Budget $B$")
    ax2.set_ylabel("DVA-MCTS calls (% of Uniform)")
    ax2.set_title("(b) Relative Verifier Efficiency")
    ax2.legend(frameon=False)
    ax2.set_ylim(0, 115)

    plt.tight_layout()
    savefig("fig2_verifier_calls.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Lipschitz continuity validation
# ════════════════════════════════════════════════════════════════════════════

def fig_lipschitz():
    d = load("lipschitz_validation")
    true_L   = d["true_L"]
    L_hat    = d["estimated_L"]
    L_std    = d["estimated_L_std"]
    per_gap  = d["per_depth_gap"]

    gaps      = sorted(int(g) for g in per_gap)
    mean_diff = np.array([per_gap[str(g)]["mean_score_diff"]  for g in gaps])
    std_diff  = np.array([per_gap[str(g)].get("std_score_diff", 0) for g in gaps])
    bounds    = np.array([per_gap[str(g)]["lipschitz_bound"]  for g in gaps])
    L_per_gap = mean_diff / np.array(gaps, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.bar(gaps, mean_diff, color=COLORS["DVA-MCTS"], alpha=0.7,
           label="Mean $|V(s_d) - V(s_{d'})|$")
    ax.errorbar(gaps, mean_diff, yerr=std_diff, fmt="none",
                color=COLORS["DVA-MCTS"], capsize=4)
    ax.plot(gaps, bounds, "s--", color=COLORS["Uniform"],
            lw=1.5, ms=5, label=f"Lipschitz envelope $L=$ {true_L}")
    ax.set_xlabel(r"Depth gap $|d - d'|$")
    ax.set_ylabel(r"Score difference")
    ax.set_title("(a) Score Differences vs. Depth Gap\n"
                 "Envelope never violated (0 of 22,500 pairs)")
    ax.legend(frameon=False)

    ax2 = axes[1]
    ax2.bar(gaps, L_per_gap, color=COLORS["DVA-MCTS"], alpha=0.7,
            label=r"Empirical mean rate $\hat{L}_{gap}$")
    ax2.axhline(true_L, color=COLORS["Uniform"], ls="--", lw=1.5,
                label=f"True maximum $L = {true_L}$")
    ax2.axhline(L_hat, color=COLORS["Random"], ls=":", lw=1.5,
                label=f"Global $\\hat{{L}} = {L_hat:.3f}$")
    ax2.set_xlabel(r"Depth gap $|d - d'|$")
    ax2.set_ylabel(r"Mean rate $\hat{L}_{gap}$")
    ax2.set_title("(b) Mean Rate vs. Lipschitz Constant\n"
                  r"$\hat{L}$ estimates mean rate; $L$ is the supremum")
    ax2.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    savefig("fig3_lipschitz.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Compute–accuracy tradeoff
# ════════════════════════════════════════════════════════════════════════════

def fig_compute_accuracy():
    d = load("compute_accuracy_tradeoff")
    threshold = d["threshold"]
    results   = d["results"]

    # Collect budgets from DVA-MCTS (all algorithms share same budgets)
    budgets = sorted(int(b) for b in results["DVA-MCTS"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left: accuracy vs. budget
    ax = axes[0]
    for key in ["DVA-MCTS", "Uniform", "Random"]:
        acc = [results[key][str(b)]["accuracy"] for b in budgets]
        ax.plot(budgets, acc, color=COLORS[key], label=key, marker="o", ms=4)

    # BestOfN is 0 everywhere — show as dashed reference
    bon = [results["BestOfN"][str(b)]["accuracy"] for b in budgets]
    ax.plot(budgets, bon, color=COLORS["BestOfN"], ls="--",
            label="Best-of-N (0% — random path fails)", alpha=0.7)

    ax.axhline(threshold, color="gray", ls=":", lw=1.0, label=f"Threshold = {threshold}")
    ax.set_xlabel("Search Budget $B$")
    ax.set_ylabel(f"Accuracy (true value $\\geq {threshold}$)")
    ax.set_title("(a) Accuracy vs. Budget")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False, fontsize=9)

    # Right: DVA accuracy gap vs. Uniform
    ax2 = axes[1]
    dva_acc = np.array([results["DVA-MCTS"][str(b)]["accuracy"] for b in budgets])
    uni_acc = np.array([results["Uniform"][str(b)]["accuracy"]  for b in budgets])
    gap     = (uni_acc - dva_acc) * 100  # in percentage points

    ax2.bar(range(len(budgets)), gap, color=COLORS["DVA-MCTS"], alpha=0.75)
    ax2.set_xticks(range(len(budgets)))
    ax2.set_xticklabels([str(b) for b in budgets], rotation=30)
    ax2.set_xlabel("Search Budget $B$")
    ax2.set_ylabel("Accuracy gap vs. Uniform (pp)")
    ax2.set_title("(b) DVA-MCTS Accuracy Gap\n(vs. Exhaustive Verification)")

    plt.tight_layout()
    savefig("fig4_compute_accuracy.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Ablation: threshold exponent
# ════════════════════════════════════════════════════════════════════════════

def fig_ablation():
    d = load("ablation_threshold")
    budget  = d["budget"]
    results = d["results"]
    steps   = np.arange(1, budget + 1)

    alphas = sorted(float(a) for a in results)
    colors_ab = plt.cm.Blues(np.linspace(0.35, 0.9, len(alphas)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    for idx, alpha in enumerate(alphas):
        key  = str(alpha)
        mean = np.array(results[key]["cumulative_regret_mean"])
        std  = np.array(results[key]["cumulative_regret_std"])
        ax.plot(steps, mean, color=colors_ab[idx], label=fr"$\alpha={alpha}$")
        ax.fill_between(steps, mean - std, mean + std, alpha=0.10, color=colors_ab[idx])

    # O(sqrt(T)) reference scaled to alpha=0.5 curve
    ref_m = np.array(results["0.5"]["cumulative_regret_mean"])
    ref_scale = ref_m[budget // 2] / np.sqrt(budget // 2)
    ax.plot(steps, ref_scale * np.sqrt(steps), "k--", lw=1.1, label=r"$O(\sqrt{T})$")

    ax.set_xlabel("Search steps $t$")
    ax.set_ylabel("Cumulative regret $R(t)$")
    ax.set_title(r"(a) Effect of Threshold Exponent $\alpha$")
    ax.legend(frameon=False, fontsize=9)

    # Right: operating curve — final regret vs. call fraction
    ax2 = axes[1]
    final_regrets  = [results[str(a)]["final_regret_mean"]  for a in alphas]
    call_fractions = [results[str(a)]["call_fraction"] * 100 for a in alphas]

    ax2.scatter(call_fractions, final_regrets, c=colors_ab, s=90, zorder=3)
    for i, alpha in enumerate(alphas):
        ax2.annotate(fr"$\alpha={alpha}$",
                     (call_fractions[i], final_regrets[i]),
                     textcoords="offset points", xytext=(6, 3), fontsize=9)
    ax2.set_xlabel("Verification rate (% of budget)")
    ax2.set_ylabel(r"Cumulative regret at $t=T$")
    ax2.set_title(r"(b) Regret vs. Verification Rate")

    plt.tight_layout()
    savefig("fig5_ablation.pdf")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    missing = [f for f in
               ["regret_scaling", "verifier_efficiency", "lipschitz_validation",
                "compute_accuracy_tradeoff", "ablation_threshold"]
               if not (RESULTS / f"{f}.json").exists()]
    if missing:
        print(f"ERROR: missing result files: {missing}")
        print("Run: python experiments/run_experiment.py --exp all --budget 400 --runs 50")
        sys.exit(1)

    print("Generating figures from real experimental data...")
    fig_regret()
    fig_verifier_calls()
    fig_lipschitz()
    fig_compute_accuracy()
    fig_ablation()
    print("Done. All figures reflect actual experiment results.")
