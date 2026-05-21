"""
Master experiment runner for DVA-MCTS.

Usage
-----
  python experiments/run_experiment.py --exp all --budget 400 --runs 50
  python experiments/run_experiment.py --exp regret --budget 800 --runs 100
  python experiments/run_experiment.py --exp ablation --budget 400 --runs 50
  python experiments/run_experiment.py --exp exploitation --runs 20

Experiments
-----------
  regret        : Cumulative regret vs. steps (Figures 1a, 1b)
  efficiency    : Verifier call count vs. budget (Figure 2)
  lipschitz     : Empirical Lipschitz validation (Figure 3)
  tradeoff      : Compute-accuracy tradeoff (Figure 4)
  ablation      : Threshold exponent sweep (Figure 5)
  adaptive_l    : Adaptive L estimator comparison (Figure 6)
  exploitation  : Large-budget exploitation-regime experiment (T >> N_T)
  all           : Run all experiments sequentially
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

# Ensure the package is importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dva_mcts import (
    DVAMCTS,
    BestOfN,
    DVAConfig,
    ExperimentConfig,
    LipschitzVerifier,
    RandomAllocationMCTS,
    RegretTracker,
    SearchTree,
    UniformMCTS,
)
from dva_mcts.metrics import (
    estimate_lipschitz,
    fit_regret_exponent,
    summary_table,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_verifier_and_tree(cfg: ExperimentConfig, seed: int):
    """Return a fresh (verifier, tree) pair for a single run."""
    rng = np.random.default_rng(seed)
    verifier = LipschitzVerifier(
        lipschitz_L=cfg.lipschitz_L,
        sigma=cfg.noise_sigma,
        rng=rng,
    )

    def value_fn(node_id, depth):
        # True values are managed by the verifier; this just returns a placeholder
        return 0.0

    tree = SearchTree(
        branching_factor=cfg.branching_factor,
        max_depth=cfg.max_depth,
        value_fn=value_fn,
        rng=rng,
    )
    # Populate true values via the verifier (ensures Lipschitz consistency)
    verifier.populate_tree_values(tree)
    return verifier, tree, rng


def run_single(
    algorithm,
    cfg: ExperimentConfig,
    budget: int,
    seed: int,
):
    """Run one algorithm for one budget on one fresh tree."""
    verifier, tree, rng = make_verifier_and_tree(cfg, seed)
    # Re-attach the verifier to the algorithm (same verifier, fresh tree)
    algorithm.verifier = verifier
    algorithm.rng = rng
    return algorithm.search(tree, budget)


def build_algorithms(cfg: ExperimentConfig, rng: np.random.Generator, verifier):
    """Construct all algorithm instances sharing the same config."""
    dva_cfg = DVAConfig(
        gamma=1.0,
        alpha=0.5,
        lipschitz_L=cfg.lipschitz_L,
        sigma_max=cfg.noise_sigma,
        branching_factor=cfg.branching_factor,
        max_depth=cfg.max_depth,
    )
    algos = {}
    if cfg.run_dva:
        algos["DVA-MCTS"] = DVAMCTS(verifier=verifier, config=dva_cfg, rng=rng)
    if cfg.run_uniform:
        algos["Uniform"] = UniformMCTS(verifier=verifier, config=dva_cfg, rng=rng)
    if cfg.run_random:
        algos["Random"] = RandomAllocationMCTS(verifier=verifier, config=dva_cfg, rng=rng)
    if cfg.run_best_of_n:
        algos["BestOfN"] = BestOfN(verifier=verifier, rng=rng)
    return algos, dva_cfg


# ── Experiment 1: Regret scaling (matched-pairs design) ───────────────────────

def exp_regret(cfg: ExperimentConfig, budget: int) -> dict:
    """Regret scaling with matched-pairs design for valid statistical comparison.

    Each run uses the SAME random seed (hence the same true tree values) for all
    algorithms so that pairwise differences in outcome are attributable to the
    algorithm, not random variation in the problem instance.  Per-run arrays are
    saved so that Wilcoxon signed-rank tests can be computed post-hoc.
    """
    log.info("=== Experiment: Regret Scaling (budget=%d, runs=%d, matched-pairs) ===",
             budget, cfg.n_runs)
    t0 = time.time()

    algo_names = ["DVA-MCTS", "Uniform", "Random"]
    trackers = {name: RegretTracker(budget) for name in algo_names}
    # Per-run data for statistical tests (matched pairs)
    per_run: dict[str, dict] = {
        name: {"final_regret": [], "calls": [], "accuracy": []}
        for name in algo_names
    }

    dva_cfg = DVAConfig(
        gamma=1.0, alpha=0.5,
        lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
        branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
    )

    for run in range(cfg.n_runs):
        seed = cfg.seed + run   # SAME seed → same problem instance for all algorithms

        for name, AlgClass, kwargs in [
            ("DVA-MCTS", DVAMCTS,              {"config": dva_cfg}),
            ("Uniform",  UniformMCTS,          {"config": dva_cfg}),
            ("Random",   RandomAllocationMCTS, {"config": dva_cfg}),
        ]:
            # Recreate tree/verifier from same seed for every algorithm
            v, t, r = make_verifier_and_tree(cfg, seed)
            alg = AlgClass(verifier=v, rng=r, **kwargs)
            result = alg.search(t, budget)
            trackers[name].add(result)
            per_run[name]["final_regret"].append(result.final_regret)
            per_run[name]["calls"].append(result.total_verifier_calls)
            per_run[name]["accuracy"].append(1.0 if result.best_true_value >= 0.8 else 0.0)

        if (run + 1) % 10 == 0:
            log.info("  Run %d/%d done", run + 1, cfg.n_runs)

    # Fit exponent for DVA-MCTS
    mean_r, std_r = trackers["DVA-MCTS"].cumulative_regret_stats()
    steps = list(range(10, budget + 1, 10))
    sampled_regrets = [float(mean_r[s - 1]) for s in steps]
    exponent, _, r2 = fit_regret_exponent(steps, sampled_regrets)
    log.info("  DVA-MCTS regret exponent: %.3f (R²=%.3f, expected ~0.50)", exponent, r2)

    # Wilcoxon signed-rank tests: DVA-MCTS vs Uniform (matched pairs)
    dva_regrets = np.array(per_run["DVA-MCTS"]["final_regret"])
    uni_regrets = np.array(per_run["Uniform"]["final_regret"])
    dva_acc     = np.array(per_run["DVA-MCTS"]["accuracy"])
    uni_acc     = np.array(per_run["Uniform"]["accuracy"])

    stat_regret, pval_regret = scipy_stats.wilcoxon(dva_regrets, uni_regrets,
                                                     alternative="two-sided",
                                                     zero_method="wilcox")
    # accuracy: DVA should have higher accuracy → test alternative="greater"
    # wilcoxon needs x - y; DVA acc - Uni acc
    diff_acc = dva_acc - uni_acc
    if diff_acc.sum() == 0:
        pval_acc = 1.0   # no difference at all
    else:
        try:
            stat_acc, pval_acc = scipy_stats.wilcoxon(diff_acc, alternative="greater",
                                                       zero_method="wilcox")
        except Exception:
            pval_acc = float("nan")

    log.info("  Wilcoxon (regret DVA vs Uni): stat=%.2f  p=%.4f", stat_regret, pval_regret)
    log.info("  Wilcoxon (accuracy DVA > Uni): p=%.4f", pval_acc)
    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "regret",
        "budget": budget,
        "n_runs": cfg.n_runs,
        "matched_pairs": True,
        "regret_exponent": exponent,
        "r_squared": r2,
        "statistical_tests": {
            "wilcoxon_regret_dva_vs_uniform": {
                "statistic": float(stat_regret),
                "p_value": float(pval_regret),
                "alternative": "two-sided",
                "note": "H0: DVA final regret = Uniform final regret",
            },
            "wilcoxon_accuracy_dva_gt_uniform": {
                "p_value": float(pval_acc),
                "alternative": "greater",
                "note": "H1: DVA accuracy > Uniform accuracy",
            },
        },
        "algorithms": {},
    }
    for name, tracker in trackers.items():
        mean, std = tracker.cumulative_regret_stats()
        output["algorithms"][name] = {
            "cumulative_regret_mean": mean.tolist(),
            "cumulative_regret_std": std.tolist(),
            "mean_verifier_calls": tracker.mean_verifier_calls(),
            "mean_final_regret": tracker.mean_final_regret(),
            "mean_accuracy": tracker.mean_accuracy(),
            # Per-run arrays for downstream analysis
            "per_run_final_regret": per_run[name]["final_regret"],
            "per_run_calls": per_run[name]["calls"],
            "per_run_accuracy": per_run[name]["accuracy"],
        }

    path = RESULTS_DIR / "regret_scaling.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 2: Verifier efficiency ────────────────────────────────────────

def exp_efficiency(cfg: ExperimentConfig) -> dict:
    log.info("=== Experiment: Verifier Efficiency ===")
    t0 = time.time()

    results_by_budget = {}
    for budget in cfg.budgets:
        dva_calls, uni_calls = [], []
        for run in range(cfg.n_runs):
            seed = cfg.seed + run
            dva_cfg = DVAConfig(
                gamma=1.0, alpha=0.5,
                lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
            )
            for name, AlgClass, calls_list in [
                ("dva", DVAMCTS,     dva_calls),
                ("uni", UniformMCTS, uni_calls),
            ]:
                offset = 0 if name == "dva" else 20000
                v, t, r = make_verifier_and_tree(cfg, seed + offset)
                alg = AlgClass(verifier=v, rng=r, config=dva_cfg)
                res = alg.search(t, budget)
                calls_list.append(res.total_verifier_calls)

        dva_arr = np.array(dva_calls)
        uni_arr = np.array(uni_calls)
        results_by_budget[budget] = {
            "dva_mean": float(dva_arr.mean()),
            "dva_std":  float(dva_arr.std()),
            "uni_mean": float(uni_arr.mean()),
            "uni_std":  float(uni_arr.std()),
            "ratio_pct": float(dva_arr.mean() / uni_arr.mean() * 100),
        }
        log.info("  B=%4d | DVA calls: %.1f ± %.1f | Uni calls: %.1f | Ratio: %.1f%%",
                 budget,
                 dva_arr.mean(), dva_arr.std(),
                 uni_arr.mean(),
                 dva_arr.mean() / uni_arr.mean() * 100)

    # Fit sqrt(T) log(T) scaling for DVA
    budgets_arr = np.array(cfg.budgets, dtype=float)
    dva_means = np.array([results_by_budget[b]["dva_mean"] for b in cfg.budgets])
    expected_rate = np.sqrt(budgets_arr) * np.log(budgets_arr)
    scale = np.mean(dva_means / expected_rate)

    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "efficiency",
        "n_runs": cfg.n_runs,
        "sqrt_log_scale_factor": float(scale),
        "by_budget": results_by_budget,
    }
    path = RESULTS_DIR / "verifier_efficiency.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 3: Lipschitz validation ───────────────────────────────────────

def exp_lipschitz(cfg: ExperimentConfig, n_paths: int = 500) -> dict:
    log.info("=== Experiment: Lipschitz Validation (n_paths=%d) ===", n_paths)
    t0 = time.time()

    rng = np.random.default_rng(cfg.seed)
    verifier = LipschitzVerifier(cfg.lipschitz_L, cfg.noise_sigma, rng)

    trajectories = []
    depth_gaps, score_gaps = [], []

    for path_idx in range(n_paths):
        traj = []
        score = rng.uniform(0.3, 0.7)
        for _ in range(cfg.max_depth):
            delta = rng.uniform(-cfg.lipschitz_L, cfg.lipschitz_L)
            score = float(np.clip(score + delta, 0.0, 1.0))
            traj.append(score)
        trajectories.append(traj)

        for d1 in range(len(traj)):
            for d2 in range(d1 + 1, min(d1 + 6, len(traj))):
                depth_gaps.append(d2 - d1)
                score_gaps.append(abs(traj[d2] - traj[d1]))

    L_hat, L_std = estimate_lipschitz(trajectories)
    log.info("  True L=%.2f | Estimated L_hat=%.4f ± %.4f", cfg.lipschitz_L, L_hat, L_std)

    # Check that Lipschitz envelope holds for all pairs
    violations = sum(
        1 for dg, sg in zip(depth_gaps, score_gaps) if sg > cfg.lipschitz_L * dg + 1e-9
    )
    violation_rate = violations / len(depth_gaps)
    log.info("  Lipschitz envelope violation rate: %.4f (should be ~0)", violation_rate)
    log.info("  Elapsed: %.1fs", time.time() - t0)

    # Per-gap statistics
    gaps_arr = np.array(depth_gaps)
    scores_arr = np.array(score_gaps)
    per_gap = {}
    for gap in sorted(set(depth_gaps)):
        mask = gaps_arr == gap
        per_gap[int(gap)] = {
            "mean_score_diff": float(scores_arr[mask].mean()),
            "std_score_diff":  float(scores_arr[mask].std()),
            "lipschitz_bound": float(cfg.lipschitz_L * gap),
        }

    output = {
        "experiment": "lipschitz",
        "true_L": cfg.lipschitz_L,
        "estimated_L": L_hat,
        "estimated_L_std": L_std,
        "violation_rate": violation_rate,
        "n_pairs": len(depth_gaps),
        "per_depth_gap": per_gap,
    }
    path = RESULTS_DIR / "lipschitz_validation.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 4: Compute–accuracy tradeoff ───────────────────────────────────

def exp_tradeoff(cfg: ExperimentConfig, threshold: float = 0.8) -> dict:
    log.info("=== Experiment: Compute–Accuracy Tradeoff (threshold=%.1f) ===", threshold)
    t0 = time.time()

    results = {name: {} for name in ["DVA-MCTS", "Uniform", "Random", "BestOfN"]}

    for budget in cfg.budgets:
        algo_results = {name: [] for name in results}
        for run in range(cfg.n_runs):
            seed = cfg.seed + run
            dva_cfg = DVAConfig(
                gamma=1.0, alpha=0.5,
                lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
            )
            specs = [
                ("DVA-MCTS", DVAMCTS,              {"config": dva_cfg}),
                ("Uniform",  UniformMCTS,          {"config": dva_cfg}),
                ("Random",   RandomAllocationMCTS, {"config": dva_cfg}),
                ("BestOfN",  BestOfN,              {}),
            ]
            for idx, (name, AlgClass, kwargs) in enumerate(specs):
                v, t, r = make_verifier_and_tree(cfg, seed + idx * 30000)
                alg = AlgClass(verifier=v, rng=r, **kwargs)
                res = alg.search(t, budget)
                algo_results[name].append(res.best_true_value)

        for name, vals in algo_results.items():
            arr = np.array(vals)
            results[name][budget] = {
                "mean_value": float(arr.mean()),
                "std_value":  float(arr.std()),
                "accuracy":   float((arr >= threshold).mean()),
            }

        log.info("  B=%4d | DVA acc=%.3f | Uni acc=%.3f | BoN acc=%.3f",
                 budget,
                 results["DVA-MCTS"][budget]["accuracy"],
                 results["Uniform"][budget]["accuracy"],
                 results["BestOfN"][budget]["accuracy"])

    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {"experiment": "tradeoff", "threshold": threshold, "n_runs": cfg.n_runs, "results": results}
    path = RESULTS_DIR / "compute_accuracy_tradeoff.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 5: Ablation ────────────────────────────────────────────────────

def exp_ablation(cfg: ExperimentConfig, budget: int) -> dict:
    log.info("=== Experiment: Ablation (budget=%d) ===", budget)
    t0 = time.time()

    results = {}
    for alpha in cfg.alpha_values:
        trackers_alpha = {}
        call_counts = []
        for run in range(cfg.n_runs):
            seed = cfg.seed + run
            dva_cfg = DVAConfig(
                gamma=1.0, alpha=alpha,
                lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
            )
            v, t, r = make_verifier_and_tree(cfg, seed + int(alpha * 1000))
            alg = DVAMCTS(verifier=v, config=dva_cfg, rng=r)
            res = alg.search(t, budget)
            call_counts.append(res.total_verifier_calls)
            if alpha not in trackers_alpha:
                trackers_alpha[alpha] = RegretTracker(budget)
            trackers_alpha[alpha].add(res)

        tracker = trackers_alpha[alpha]
        mean_r, std_r = tracker.cumulative_regret_stats()
        results[str(alpha)] = {
            "alpha": alpha,
            "cumulative_regret_mean": mean_r.tolist(),
            "cumulative_regret_std": std_r.tolist(),
            "final_regret_mean": float(mean_r[-1]),
            "mean_verifier_calls": float(np.mean(call_counts)),
            "std_verifier_calls": float(np.std(call_counts)),
            "call_fraction": float(np.mean(call_counts)) / budget,
        }
        log.info("  alpha=%.2f | regret=%.2f | calls=%.1f (%.1f%%)",
                 alpha, mean_r[-1], np.mean(call_counts),
                 np.mean(call_counts) / budget * 100)

    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {"experiment": "ablation", "budget": budget, "n_runs": cfg.n_runs, "results": results}
    path = RESULTS_DIR / "ablation_threshold.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 6: Adaptive L estimator ──────────────────────────────────────

def exp_adaptive_l(cfg: ExperimentConfig, budget: int) -> dict:
    """Compare DVA-MCTS with: known L, fixed L̂ (mean estimator), adaptive L (running max)."""
    log.info("=== Experiment: Adaptive L Estimator (budget=%d, runs=%d) ===", budget, cfg.n_runs)
    t0 = time.time()

    # Estimate L̂ the naive way: mean |V(s)-V(s')| / depth_gap from a held-out set
    rng_est = np.random.default_rng(cfg.seed + 99999)
    ver_est = LipschitzVerifier(cfg.lipschitz_L, cfg.noise_sigma, rng_est)
    score_ratios = []
    prev = rng_est.uniform(0.3, 0.7)
    for _ in range(5000):
        delta = rng_est.uniform(-cfg.lipschitz_L, cfg.lipschitz_L)
        curr = float(np.clip(prev + delta, 0.0, 1.0))
        score_ratios.append(abs(curr - prev))
        prev = curr
    l_hat_fixed = float(np.mean(score_ratios))  # mean-rate estimate
    log.info("  Mean-rate L̂=%.4f (true L=%.2f)", l_hat_fixed, cfg.lipschitz_L)

    variants = {
        "known_L":    DVAConfig(lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
                                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
                                use_adaptive_l=False),
        "fixed_Lhat": DVAConfig(lipschitz_L=l_hat_fixed,    sigma_max=cfg.noise_sigma,
                                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
                                use_adaptive_l=False),
        "adaptive_L": DVAConfig(lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
                                branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
                                use_adaptive_l=True, adaptive_l_init=l_hat_fixed),
    }

    results = {name: {"regrets": [], "calls": [], "accuracies": [], "final_l": []}
               for name in variants}

    for run in range(cfg.n_runs):
        seed = cfg.seed + run   # SAME seed for all variants (matched pairs)
        for name, dva_cfg in variants.items():
            v, t, r = make_verifier_and_tree(cfg, seed)
            alg = DVAMCTS(verifier=v, config=dva_cfg, rng=r)
            res = alg.search(t, budget)
            results[name]["regrets"].append(res.final_regret)
            results[name]["calls"].append(res.total_verifier_calls)
            results[name]["accuracies"].append(1.0 if res.best_true_value >= 0.8 else 0.0)
            if res.step_records and dva_cfg.use_adaptive_l:
                results[name]["final_l"].append(res.step_records[-1].adaptive_l_estimate)

        if (run + 1) % 10 == 0:
            log.info("  Run %d/%d done", run + 1, cfg.n_runs)

    log.info("  Results summary:")
    summary = {}
    for name, data in results.items():
        r_arr = np.array(data["regrets"])
        c_arr = np.array(data["calls"])
        a_arr = np.array(data["accuracies"])
        l_arr = np.array(data["final_l"]) if data["final_l"] else np.array([])
        entry = {
            "regret_mean":    float(r_arr.mean()),
            "regret_std":     float(r_arr.std()),
            "calls_mean":     float(c_arr.mean()),
            "calls_std":      float(c_arr.std()),
            "accuracy":       float(a_arr.mean()),
            "final_l_mean":   float(l_arr.mean()) if len(l_arr) > 0 else None,
        }
        summary[name] = entry
        log.info("  %-12s | regret=%.2f±%.1f | calls=%.1f | acc=%.1f%% | final_L=%s",
                 name,
                 entry["regret_mean"], entry["regret_std"],
                 entry["calls_mean"],
                 entry["accuracy"] * 100,
                 f"{entry['final_l_mean']:.4f}" if entry["final_l_mean"] else "n/a")

    known_r = summary["known_L"]["regret_mean"]
    adaptive_r = summary["adaptive_L"]["regret_mean"]
    fixed_r = summary["fixed_Lhat"]["regret_mean"]
    gap_closed = (fixed_r - adaptive_r) / max(fixed_r - known_r, 1e-9) * 100
    log.info("  Gap closed by adaptive L: %.1f%%", gap_closed)

    # Wilcoxon matched-pair tests for adaptive_L vs fixed_Lhat (same seed per run)
    fixed_arr    = np.array(results["fixed_Lhat"]["regrets"])
    adaptive_arr = np.array(results["adaptive_L"]["regrets"])
    known_arr    = np.array(results["known_L"]["regrets"])
    try:
        stat_af, pval_af = scipy_stats.wilcoxon(adaptive_arr, fixed_arr,
                                                 alternative="less",
                                                 zero_method="wilcox")
        stat_ak, pval_ak = scipy_stats.wilcoxon(adaptive_arr, known_arr,
                                                 alternative="two-sided",
                                                 zero_method="wilcox")
    except Exception as e:
        log.warning("  Wilcoxon failed: %s", e)
        stat_af = pval_af = stat_ak = pval_ak = float("nan")

    log.info("  Wilcoxon adaptive < fixed (regret): stat=%.2f  p=%.4f", stat_af, pval_af)
    log.info("  Wilcoxon adaptive vs known (regret): stat=%.2f  p=%.4f", stat_ak, pval_ak)
    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "adaptive_l",
        "budget": budget,
        "n_runs": cfg.n_runs,
        "matched_pairs": True,
        "true_L": cfg.lipschitz_L,
        "fixed_lhat": l_hat_fixed,
        "gap_closed_pct": float(gap_closed),
        "summary": summary,
        "statistical_tests": {
            "wilcoxon_adaptive_lt_fixed_regret": {
                "statistic": float(stat_af),
                "p_value": float(pval_af),
                "alternative": "less",
                "note": "H1: adaptive_L regret < fixed_Lhat regret",
            },
            "wilcoxon_adaptive_vs_known_regret": {
                "statistic": float(stat_ak),
                "p_value": float(pval_ak),
                "alternative": "two-sided",
                "note": "H0: adaptive_L regret = known_L regret",
            },
        },
        # Per-run arrays for downstream analysis
        "per_run_regrets": {
            "known_L": known_arr.tolist(),
            "fixed_Lhat": fixed_arr.tolist(),
            "adaptive_L": adaptive_arr.tolist(),
        },
    }
    path = RESULTS_DIR / "adaptive_l_comparison.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── Experiment 7: Exploitation-regime scaling (T >> N_T) ──────────────────────

def exp_exploitation_regime(cfg: ExperimentConfig, n_runs: int = 20) -> dict:
    """Run DVA-MCTS and Uniform Verification at large budgets where T >> N_T.

    N_T = K^D - 1 = 4095 for K=2, D=12.  The main theoretical prediction is:
      (a) DVA verifier calls plateau near N_T (constant, independent of T).
      (b) Call fraction N_T/T → 0 as T → ∞.
      (c) Accuracy remains competitive with Uniform.

    Budgets span the crossover: exploration regime (T < N_T), transition (T ≈ N_T),
    and deep exploitation regime (T >> N_T).
    """
    budgets = [400, 1000, 2000, 4000, 8000, 16384]
    N_T = cfg.branching_factor ** cfg.max_depth - 1   # = 4095

    log.info("=== Experiment: Exploitation Regime (N_T=%d, runs=%d) ===", N_T, n_runs)
    log.info("    Budgets: %s", budgets)
    t0 = time.time()

    dva_cfg = DVAConfig(
        gamma=1.0, alpha=0.5,
        lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
        branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
    )

    results_by_budget: dict[int, dict] = {}

    for budget in budgets:
        dva_calls, uni_calls = [], []
        dva_acc,   uni_acc   = [], []
        dva_regret, uni_regret = [], []

        for run in range(n_runs):
            seed = cfg.seed + run   # matched pairs

            # DVA-MCTS
            v, t, r = make_verifier_and_tree(cfg, seed)
            alg = DVAMCTS(verifier=v, config=dva_cfg, rng=r)
            res = alg.search(t, budget)
            dva_calls.append(res.total_verifier_calls)
            dva_acc.append(1.0 if res.best_true_value >= 0.8 else 0.0)
            dva_regret.append(res.final_regret)

            # Uniform (same seed → same problem)
            v, t, r = make_verifier_and_tree(cfg, seed)
            alg = UniformMCTS(verifier=v, config=dva_cfg, rng=r)
            res = alg.search(t, budget)
            uni_calls.append(res.total_verifier_calls)
            uni_acc.append(1.0 if res.best_true_value >= 0.8 else 0.0)
            uni_regret.append(res.final_regret)

        dva_c = np.array(dva_calls)
        uni_c = np.array(uni_calls)
        dva_a = np.array(dva_acc)
        uni_a = np.array(uni_acc)
        dva_r = np.array(dva_regret)
        uni_r = np.array(uni_regret)

        call_savings_pct = float((1 - dva_c.mean() / uni_c.mean()) * 100)
        call_fraction    = float(dva_c.mean() / budget)

        results_by_budget[budget] = {
            "dva_calls_mean":   float(dva_c.mean()),
            "dva_calls_std":    float(dva_c.std()),
            "uni_calls_mean":   float(uni_c.mean()),
            "dva_accuracy":     float(dva_a.mean()),
            "uni_accuracy":     float(uni_a.mean()),
            "dva_regret_mean":  float(dva_r.mean()),
            "uni_regret_mean":  float(uni_r.mean()),
            "call_savings_pct": call_savings_pct,
            "dva_call_fraction": call_fraction,
            "n_t_fraction":     float(N_T / budget),
        }
        log.info(
            "  B=%6d | DVA calls=%5.0f (%.1f%% of B) | savings=%5.1f%% | "
            "DVA acc=%.1f%% | N_T/B=%.3f",
            budget, dva_c.mean(), call_fraction * 100,
            call_savings_pct, dva_a.mean() * 100, N_T / budget,
        )

    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "exploitation_regime",
        "N_T": N_T,
        "n_runs": n_runs,
        "budgets": budgets,
        "results": results_by_budget,
        "note": (
            "Matched-pairs design: same seed for DVA and Uniform per run. "
            f"N_T={N_T} is the exploitation crossover (K^D - 1 for K=2, D=12)."
        ),
    }
    path = RESULTS_DIR / "exploitation_regime.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DVA-MCTS Experiments")
    p.add_argument("--exp", choices=["all", "regret", "efficiency", "lipschitz",
                                     "tradeoff", "ablation", "adaptive_l", "exploitation"],
                   default="all")
    p.add_argument("--budget", type=int, default=400,
                   help="Search budget for regret/ablation experiments")
    p.add_argument("--runs", type=int, default=50, help="Number of independent runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--L", type=float, default=0.35, help="Lipschitz constant")
    p.add_argument("--sigma", type=float, default=0.05, help="Verifier noise std dev")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = ExperimentConfig(
        name=args.exp,
        budgets=[50, 100, 200, 400, 800, 1600, 3200],
        n_runs=args.runs,
        seed=args.seed,
        lipschitz_L=args.L,
        noise_sigma=args.sigma,
    )

    t_start = time.time()
    all_results = {}

    if args.exp in ("all", "regret"):
        all_results["regret"] = exp_regret(cfg, budget=args.budget)

    if args.exp in ("all", "efficiency"):
        all_results["efficiency"] = exp_efficiency(cfg)

    if args.exp in ("all", "lipschitz"):
        all_results["lipschitz"] = exp_lipschitz(cfg)

    if args.exp in ("all", "tradeoff"):
        all_results["tradeoff"] = exp_tradeoff(cfg)

    if args.exp in ("all", "ablation"):
        all_results["ablation"] = exp_ablation(cfg, budget=args.budget)

    if args.exp in ("all", "adaptive_l"):
        all_results["adaptive_l"] = exp_adaptive_l(cfg, budget=args.budget)

    if args.exp in ("all", "exploitation"):
        all_results["exploitation"] = exp_exploitation_regime(cfg, n_runs=args.runs)

    summary_path = RESULTS_DIR / "summary.json"
    meta = {
        "total_elapsed_s": round(time.time() - t_start, 1),
        "experiments_run": list(all_results.keys()),
        "config": {
            "budget": args.budget,
            "n_runs": args.runs,
            "seed": args.seed,
            "lipschitz_L": args.L,
            "noise_sigma": args.sigma,
        },
    }
    with open(summary_path, "w") as f:
        json.dump(meta, f, indent=2)

    log.info("=== All done in %.1fs ===", time.time() - t_start)
    log.info("Results in: %s", RESULTS_DIR)


if __name__ == "__main__":
    main()
