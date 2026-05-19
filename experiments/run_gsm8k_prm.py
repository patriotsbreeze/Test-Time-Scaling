"""
Real PRM experiment: DVA-MCTS vs Uniform on GSM8K with Math-Shepherd step scores.

Uses the peiyi9979/math-shepherd dataset (Lightman et al. 2023 / Wang et al. 2024)
which provides binary process reward model scores for each step of GSM8K solutions.

Each problem has K=4 candidate solutions. The MCTS algorithm identifies the
highest-quality solution while minimizing verifier calls. We validate that:
  1. Real PRM scores approximately satisfy the Lipschitz condition.
  2. DVA-MCTS achieves fewer verifier calls than Uniform at comparable accuracy.

Usage
-----
  python experiments/run_gsm8k_prm.py --problems 100 --budget 400 --runs 20
  python experiments/run_gsm8k_prm.py --problems 100 --budget 400 --runs 20 --download
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATASET_PATH = RESULTS_DIR / "gsm8k_prm_dataset.json"


# ── Data loading ─────────────────────────────────────────────────────────────

def download_dataset(max_rows: int = 30000) -> None:
    """Download and parse Math-Shepherd GSM8K data."""
    import re
    import collections

    try:
        from datasets import load_dataset
    except ImportError:
        log.error("Install huggingface datasets: python3 -m pip install datasets --user")
        sys.exit(1)

    log.info("Downloading Math-Shepherd GSM8K data (may take a few minutes)...")
    ds = load_dataset("peiyi9979/math-shepherd", split="train", streaming=True)

    gsm_rows = []
    for row in ds:
        if row["task"] == "GSM8K":
            gsm_rows.append(row)
        if len(gsm_rows) >= max_rows:
            break
    log.info("  Collected %d GSM8K rows", len(gsm_rows))

    def parse_solution(label_text: str) -> Tuple[str, List[float]]:
        split_idx = label_text.find("Step 1:")
        if split_idx == -1:
            return "", []
        problem = label_text[:split_idx].strip()
        steps_text = label_text[split_idx:]
        step_labels: List[float] = []
        for m in re.finditer(r"Step \d+: .*? ([+-])(?=\n|$)", steps_text, re.DOTALL):
            step_labels.append(1.0 if m.group(1) == "+" else 0.0)
        return problem, step_labels

    by_problem: Dict[str, List[List[float]]] = collections.defaultdict(list)
    for row in gsm_rows:
        prob, steps = parse_solution(row["label"])
        if prob and len(steps) >= 2:
            by_problem[prob].append(steps)

    # Keep problems with multiple solutions and quality variation
    useful = []
    for prob, solutions in by_problem.items():
        if len(solutions) < 2:
            continue
        qualities = [np.mean(s) for s in solutions]
        q_range = max(qualities) - min(qualities)
        if q_range > 0.1:
            useful.append({"problem": prob, "solutions": solutions, "n_solutions": len(solutions)})

    # Normalise to K=4 solutions, depth truncated to shortest solution
    dataset = []
    for p in useful[:200]:
        sols = p["solutions"][:4]
        while len(sols) < 4:  # pad with copies if fewer than 4
            sols.append(sols[0])
        d = min(min(len(s) for s in sols), 8)
        dataset.append({
            "problem": p["problem"][:200],
            "solutions": [s[:d] for s in sols],
            "depth": d,
        })

    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f)
    log.info("  Saved %d problems → %s", len(dataset), DATASET_PATH)


def load_dataset_local() -> List[dict]:
    if not DATASET_PATH.exists():
        log.info("Dataset not found; downloading...")
        download_dataset()
    with open(DATASET_PATH) as f:
        return json.load(f)


# ── Per-problem MCTS simulation ───────────────────────────────────────────────

class _ProblemState:
    """Lightweight per-problem state for the K-path MCTS simulation."""

    def __init__(self, solutions: List[List[float]], sigma: float, rng: np.random.Generator):
        self.solutions = solutions  # K × D
        self.K = len(solutions)
        self.D = max(len(s) for s in solutions)
        self.sigma = sigma
        self.rng = rng

        # per-path state
        self.verified: Dict[Tuple[int, int], float] = {}   # (k, d) -> observed score
        self.is_proxy:  Dict[Tuple[int, int], bool]  = {}  # True if proxy, not real call
        self.visit_count = np.zeros(self.K)

    def true_score(self, k: int, d: int) -> float:
        sols = self.solutions[k]
        return sols[d] if d < len(sols) else sols[-1]

    def observe(self, k: int, d: int) -> float:
        """Call the verifier: return noisy score, record it, increment counter."""
        v = float(np.clip(self.true_score(k, d) + self.rng.normal(0, self.sigma), 0.0, 1.0))
        self.verified[(k, d)] = v
        self.is_proxy[(k, d)] = False
        return v  # returned so callers can update adaptive L

    def apply_proxy(self, k: int, d: int, ancestor_score: float) -> None:
        """Store Lipschitz proxy without calling verifier."""
        self.verified[(k, d)] = ancestor_score
        self.is_proxy[(k, d)] = True

    def estimated_quality(self, k: int) -> float:
        """Mean of all known scores for solution k."""
        scores = [v for (kk, dd), v in self.verified.items() if kk == k]
        if not scores:
            return 0.5  # neutral prior
        return float(np.mean(scores))

    def nearest_verified_ancestor(self, k: int, d: int) -> Optional[Tuple[int, float]]:
        """Return (depth, score) of deepest REAL verified step before d on path k."""
        for dd in range(d - 1, -1, -1):
            if (k, dd) in self.verified and not self.is_proxy.get((k, dd), True):
                return dd, self.verified[(k, dd)]
        return None

    def next_unverified_depth(self, k: int) -> Optional[int]:
        for d in range(self.D):
            if (k, d) not in self.verified:
                return d
        return None  # all steps explored


def _run_dva(
    problem: dict,
    budget: int,
    L: float,
    sigma: float,
    alpha: float,
    gamma: float,
    c_ucb: float,
    rng: np.random.Generator,
    uniform: bool = False,
    adaptive_l: bool = False,
    adaptive_l_init: float = 0.1,
) -> Tuple[int, float, float]:
    """
    Run DVA-MCTS (or Uniform if uniform=True) on one problem.

    Returns
    -------
    verifier_calls : int
    accuracy       : float  (1.0 if best-quality solution found)
    true_best_mean : float  mean true score of selected solution
    """
    solutions = problem["solutions"]
    K = len(solutions)
    state = _ProblemState(solutions, sigma, rng)
    verifier_calls = 0
    current_L = adaptive_l_init if adaptive_l else L  # running max estimate

    for t in range(1, budget + 1):
        tau = gamma / t ** alpha

        # ── Selection: UCT over K solution branches ───────────────────────
        n_total = state.visit_count.sum()
        estimates = np.array([state.estimated_quality(k) for k in range(K)])
        exploration = c_ucb * np.sqrt(np.log(n_total + 1) / (state.visit_count + 1))
        uct_scores = estimates + exploration
        k = int(np.argmax(uct_scores))
        state.visit_count[k] += 1

        # ── Find next step to explore in solution k ───────────────────────
        next_d = state.next_unverified_depth(k)
        if next_d is None:
            continue  # solution k fully explored; UCT will redirect

        # ── Verifier decision ─────────────────────────────────────────────
        if uniform:
            state.observe(k, next_d)
            verifier_calls += 1
        else:
            ancestor = state.nearest_verified_ancestor(k, next_d)
            if ancestor is None:
                state.observe(k, next_d)
                verifier_calls += 1
                # Seed adaptive L with root score change if possible
            else:
                anc_d, anc_score = ancestor
                depth_gap = next_d - anc_d
                delta = current_L * depth_gap
                if delta > tau:
                    new_score = state.observe(k, next_d)
                    verifier_calls += 1
                    if adaptive_l:
                        ratio = abs(new_score - anc_score) / depth_gap
                        if ratio > current_L:
                            current_L = ratio
                else:
                    state.apply_proxy(k, next_d, anc_score)

    # ── Evaluate ──────────────────────────────────────────────────────────
    est_quals = np.array([state.estimated_quality(k) for k in range(K)])
    selected_k = int(np.argmax(est_quals))

    # True quality = mean step score
    true_quals = np.array([np.mean(s) for s in solutions])
    oracle_k = int(np.argmax(true_quals))
    accuracy = float(selected_k == oracle_k)
    true_best = float(true_quals[selected_k])

    return verifier_calls, accuracy, true_best


# ── Lipschitz validation on real PRM data ────────────────────────────────────

def validate_lipschitz(dataset: List[dict]) -> dict:
    """Estimate empirical Lipschitz constant from real PRM scores."""
    ratios = []
    for prob in dataset:
        for sol in prob["solutions"]:
            for d1 in range(len(sol)):
                for d2 in range(d1 + 1, len(sol)):
                    gap = d2 - d1
                    score_diff = abs(sol[d2] - sol[d1])
                    ratios.append(score_diff / gap)

    arr = np.array(ratios)
    return {
        "n_pairs": len(arr),
        "empirical_L_mean": float(arr.mean()),
        "empirical_L_std": float(arr.std()),
        "empirical_L_max": float(arr.max()),
        "empirical_L_p95": float(np.percentile(arr, 95)),
        "empirical_L_p99": float(np.percentile(arr, 99)),
    }


# ── Main experiment ──────────────────────────────────────────────────────────

def run_gsm8k_experiment(
    n_problems: int,
    budget: int,
    n_runs: int,
    L: float,
    sigma: float,
    alpha: float,
    gamma: float,
    seed: int,
) -> dict:
    dataset = load_dataset_local()
    problems = dataset[:n_problems]

    log.info("=== GSM8K PRM Experiment ===")
    log.info("  Problems: %d | Budget: %d | Runs: %d | L=%.2f | sigma=%.2f",
             len(problems), budget, n_runs, L, sigma)

    # Validate Lipschitz property on real data
    lip_stats = validate_lipschitz(problems)
    log.info("  Real PRM Lipschitz stats: mean=%.3f max=%.3f p95=%.3f p99=%.3f",
             lip_stats["empirical_L_mean"], lip_stats["empirical_L_max"],
             lip_stats["empirical_L_p95"], lip_stats["empirical_L_p99"])

    records: Dict[str, dict] = {
        "DVA-MCTS":  {"calls": [], "acc": [], "kwargs": {"uniform": False, "adaptive_l": False}},
        "Uniform":   {"calls": [], "acc": [], "kwargs": {"uniform": True}},
    }

    t0 = time.time()
    for run in range(n_runs):
        for variant, rec in records.items():
            run_calls, run_acc = [], []
            for prob in problems:
                rng_v = np.random.default_rng(seed + run * 10000
                                              + hash(prob["problem"]) % 10000
                                              + abs(hash(variant)) % 5000)
                calls, acc, _ = _run_dva(
                    prob, budget, L, sigma, alpha, gamma, c_ucb=1.414,
                    rng=rng_v, **rec["kwargs"])
                run_calls.append(calls)
                run_acc.append(acc)
            rec["calls"].append(float(np.mean(run_calls)))
            rec["acc"].append(float(np.mean(run_acc)))

        if (run + 1) % 5 == 0:
            log.info("  Run %d/%d | DVA: calls=%.1f acc=%.1f%% | Uni: calls=%.1f acc=%.1f%%",
                     run + 1, n_runs,
                     np.mean(records["DVA-MCTS"]["calls"]), np.mean(records["DVA-MCTS"]["acc"]) * 100,
                     np.mean(records["Uniform"]["calls"]), np.mean(records["Uniform"]["acc"]) * 100)

    log.info("  === Summary ===")
    summary = {}
    for variant, rec in records.items():
        c = np.array(rec["calls"])
        a = np.array(rec["acc"])
        log.info("  %-12s: calls=%.1f±%.1f  accuracy=%.1f%%±%.1f%%",
                 variant, c.mean(), c.std(), a.mean()*100, a.std()*100)
        summary[variant] = {
            "mean_calls": float(c.mean()), "std_calls": float(c.std()),
            "mean_accuracy": float(a.mean()), "std_accuracy": float(a.std()),
        }

    uni_c = np.mean(records["Uniform"]["calls"])
    dva_c = np.mean(records["DVA-MCTS"]["calls"])
    log.info("  DVA call reduction: %.1f%%", (1 - dva_c/uni_c) * 100)
    log.info("  Elapsed: %.1fs", time.time() - t0)

    output = {
        "experiment": "gsm8k_prm",
        "n_problems": len(problems),
        "budget": budget,
        "n_runs": n_runs,
        "config": {"L": L, "sigma": sigma, "alpha": alpha, "gamma": gamma},
        "lipschitz_validation": lip_stats,
        "results": summary,
        "dva_call_reduction_pct": float((1 - dva_c / uni_c) * 100),
    }

    path = RESULTS_DIR / "gsm8k_prm_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("  Saved → %s", path)
    return output


def parse_args():
    p = argparse.ArgumentParser(description="GSM8K Real PRM Experiment")
    p.add_argument("--problems", type=int, default=100, help="Number of GSM8K problems")
    p.add_argument("--budget",   type=int, default=400, help="Search budget per problem")
    p.add_argument("--runs",     type=int, default=20,  help="Independent runs (different noise seeds)")
    p.add_argument("--L",        type=float, default=0.50, help="Lipschitz constant estimate (0.5 empirically for real PRM)")
    p.add_argument("--sigma",    type=float, default=0.05, help="Verifier noise std dev")
    p.add_argument("--alpha",    type=float, default=0.5,  help="Threshold exponent")
    p.add_argument("--gamma",    type=float, default=1.0,  help="Threshold scale")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--download", action="store_true", help="Force re-download dataset")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.download or not DATASET_PATH.exists():
        download_dataset()
    run_gsm8k_experiment(
        n_problems=args.problems,
        budget=args.budget,
        n_runs=args.runs,
        L=args.L,
        sigma=args.sigma,
        alpha=args.alpha,
        gamma=args.gamma,
        seed=args.seed,
    )
