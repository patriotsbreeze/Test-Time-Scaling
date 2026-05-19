"""
Evaluation metrics for DVA-MCTS experiments.

  - RegretTracker: accumulates per-step regret across runs
  - compute_accuracy: fraction of runs where solution quality > threshold
  - compute_verifier_efficiency: call reduction vs. uniform baseline
  - estimate_lipschitz: empirical Lipschitz constant from PRM trajectories
  - fit_regret_exponent: log-log slope of cumulative regret (should be ~0.5)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from .algorithm import SearchResult, StepRecord


class RegretTracker:
    """
    Accumulates regret and verifier-call statistics across multiple runs.

    Usage
    -----
    tracker = RegretTracker(budget=400)
    for result in results:
        tracker.add(result)
    mean_regret, std_regret = tracker.cumulative_regret_stats()
    """

    def __init__(self, budget: int) -> None:
        self.budget = budget
        self._regrets: List[np.ndarray] = []
        self._call_counts: List[int] = []
        self._final_values: List[float] = []
        self._oracle_values: List[float] = []

    def add(self, result: SearchResult) -> None:
        steps = result.total_steps
        regret_series = np.zeros(steps)
        for rec in result.step_records:
            if rec.step <= steps:
                regret_series[rec.step - 1] = rec.cumulative_regret
        # Forward-fill if step_records are sparse
        for i in range(1, steps):
            if regret_series[i] == 0 and regret_series[i - 1] > 0:
                regret_series[i] = regret_series[i - 1]
        self._regrets.append(regret_series)
        self._call_counts.append(result.total_verifier_calls)
        self._final_values.append(result.best_true_value)
        self._oracle_values.append(result.oracle_best_true_value)

    def cumulative_regret_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) of cumulative regret over time."""
        if not self._regrets:
            return np.zeros(self.budget), np.zeros(self.budget)
        arr = np.stack(self._regrets, axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    def mean_verifier_calls(self) -> float:
        return float(np.mean(self._call_counts)) if self._call_counts else 0.0

    def std_verifier_calls(self) -> float:
        return float(np.std(self._call_counts)) if self._call_counts else 0.0

    def mean_final_regret(self) -> float:
        regrets = [r[-1] for r in self._regrets]
        return float(np.mean(regrets)) if regrets else 0.0

    def mean_accuracy(self, threshold: float = 0.8) -> float:
        if not self._final_values:
            return 0.0
        return float(np.mean([v >= threshold for v in self._final_values]))

    def n_runs(self) -> int:
        return len(self._regrets)


def compute_accuracy(results: List[SearchResult], threshold: float = 0.8) -> Tuple[float, float]:
    """Return (mean accuracy, std) across results."""
    accs = [float(r.best_true_value >= threshold) for r in results]
    return float(np.mean(accs)), float(np.std(accs))


def compute_verifier_efficiency(
    dva_results: List[SearchResult],
    uniform_results: List[SearchResult],
) -> Tuple[float, float]:
    """
    Return the mean fraction of verifier calls used by DVA-MCTS relative
    to Uniform Verification. Values < 1.0 indicate efficiency gains.
    """
    dva_calls = np.array([r.total_verifier_calls for r in dva_results], dtype=float)
    uni_calls = np.array([r.total_verifier_calls for r in uniform_results], dtype=float)
    ratios = dva_calls / np.maximum(uni_calls, 1.0)
    return float(ratios.mean()), float(ratios.std())


def estimate_lipschitz(score_trajectories: List[List[float]]) -> Tuple[float, float]:
    """
    Empirically estimate the Lipschitz constant L from observed score trajectories.

    Parameters
    ----------
    score_trajectories : list of lists
        Each inner list is the sequence of verifier scores along one root-to-leaf path.

    Returns
    -------
    L_hat : float
        Estimated Lipschitz constant.
    L_std : float
        Standard deviation of per-pair estimates.
    """
    estimates = []
    for traj in score_trajectories:
        for i in range(len(traj)):
            for j in range(i + 1, min(i + 6, len(traj))):
                depth_gap = j - i
                score_gap = abs(traj[j] - traj[i])
                estimates.append(score_gap / max(depth_gap, 1))

    if not estimates:
        return 0.0, 0.0
    return float(np.mean(estimates)), float(np.std(estimates))


def fit_regret_exponent(
    budgets: List[int],
    mean_regrets: List[float],
) -> Tuple[float, float, float]:
    """
    Fit log(R) = exponent * log(T) + const via OLS on log-log scale.

    Returns (exponent, intercept, r_squared).
    Expected exponent is ~0.5 for the O(sqrt(T)) rate.
    """
    log_t = np.log(budgets)
    log_r = np.log(np.maximum(mean_regrets, 1e-10))
    slope, intercept, r_value, _, _ = stats.linregress(log_t, log_r)
    return float(slope), float(intercept), float(r_value ** 2)


def summary_table(
    algorithm_names: List[str],
    trackers: List[RegretTracker],
    budget: int,
    threshold: float = 0.8,
) -> Dict:
    """Produce a summary dict suitable for printing or saving as JSON."""
    rows = {}
    for name, tracker in zip(algorithm_names, trackers):
        rows[name] = {
            "mean_final_regret": round(tracker.mean_final_regret(), 4),
            "mean_verifier_calls": round(tracker.mean_verifier_calls(), 1),
            "std_verifier_calls": round(tracker.std_verifier_calls(), 1),
            "mean_accuracy": round(tracker.mean_accuracy(threshold), 4),
            "n_runs": tracker.n_runs(),
        }
    return rows
