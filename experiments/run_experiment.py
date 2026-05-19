"""
Master experiment runner for DVA-MCTS.

Usage
-----
  python experiments/run_experiment.py --exp all --budget 400 --runs 50
  python experiments/run_experiment.py --exp regret --budget 800 --runs 100
  python experiments/run_experiment.py --exp ablation --budget 400 --runs 50

Experiments
-----------
  regret      : Cumulative regret vs. steps (Figures 1a, 1b)
  efficiency  : Verifier call count vs. budget (Figure 2)
  lipschitz   : Empirical Lipschitz validation (Figure 3)
  tradeoff    : Compute-accuracy tradeoff (Figure 4)
  ablation    : Threshold exponent sweep (Figure 5)
  all         : Run all experiments sequentially
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


# ── Experiment 1: Regret scaling ─────────────────────────────────────────────

def exp_regret(cfg: ExperimentConfig, budget: int) -> dict:
    log.info("=== Experiment: Regret Scaling (budget=%d, runs=%d) ===", budget, cfg.n_runs)
    t0 = time.time()

    trackers = {name: RegretTracker(budget) for name in ["DVA-MCTS", "Uniform", "Random"]}

    for run in range(cfg.n_runs):
        seed = cfg.seed + run
        rng = np.random.default_rng(seed)
        verifier, tree, rng = make_verifier_and_tree(cfg, seed)
        dva_cfg = DVAConfig(
            gamma=1.0, alpha=0.5,
            lipschitz_L=cfg.lipschitz_L, sigma_max=cfg.noise_sigma,
            branching_factor=cfg.branching_factor, max_depth=cfg.max_depth,
        )

        for name, AlgClass, kwargs in [
            ("DVA-MCTS", DVAMCTS,              {"config": dva_cfg}),
            ("Uniform",  UniformMCTS,          {"config": dva_cfg}),
            ("Random",   RandomAllocationMCTS, {"config": dva_cfg}),
        ]:
            v2, t2, r2 = make_verifier_and_tree(cfg, seed + 10000 * list(trackers).index(name))
            alg = AlgClass(verifier=v2, rng=r2, **kwargs)
            result = alg.search(t2, budget)
            trackers[name].add(result)

        if (run + 1) % 10 == 0:
            log.info("  Run %d/%d done", run + 1, cfg.n_runs)

    # Fit exponent for DVA-MCTS
    mean_r, std_r = trackers["DVA-MCTS"].cumulative_regret_stats()
    steps = list(range(10, budget + 1, 10))
    sampled_regrets = [float(mean_r[s - 1]) for s in steps]
    exponent, _, r2 = fit_regret_exponent(steps, sampled_regrets)

    log.info("  DVA-MCTS regret exponent: %.3f (R²=%.3f, expected ~0.50)", exponent, r2)
    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "regret",
        "budget": budget,
        "n_runs": cfg.n_runs,
        "regret_exponent": exponent,
        "r_squared": r2,
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DVA-MCTS Experiments")
    p.add_argument("--exp", choices=["all", "regret", "efficiency", "lipschitz", "tradeoff", "ablation"],
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
