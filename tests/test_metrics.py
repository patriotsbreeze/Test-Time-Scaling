"""Unit tests for evaluation metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dva_mcts.algorithm import SearchResult, StepRecord
from dva_mcts.metrics import (
    RegretTracker,
    compute_accuracy,
    compute_verifier_efficiency,
    estimate_lipschitz,
    fit_regret_exponent,
)


def make_result(final_regret=1.0, calls=50, true_value=0.85, oracle=0.95, budget=100):
    steps = []
    for t in range(1, budget + 1):
        steps.append(StepRecord(
            step=t, node_id=t, depth=t % 5,
            verifier_called=(t % 2 == 0),
            estimated_value=0.7, true_value=true_value,
            oracle_best_true=oracle,
            cumulative_regret=final_regret * t / budget,
            total_verifier_calls=calls * t // budget,
        ))
    return SearchResult(
        best_node_id=0, best_estimated_value=0.8,
        best_true_value=true_value, oracle_best_true_value=oracle,
        total_steps=budget, total_verifier_calls=calls,
        verifier_call_fraction=calls / budget,
        step_records=steps,
    )


class TestRegretTracker:
    def test_empty(self):
        tracker = RegretTracker(100)
        mean, std = tracker.cumulative_regret_stats()
        assert mean.shape == (100,)
        assert std.shape == (100,)
        assert tracker.mean_final_regret() == 0.0

    def test_single_run(self):
        tracker = RegretTracker(100)
        result = make_result(final_regret=5.0, calls=40, true_value=0.85)
        tracker.add(result)
        assert tracker.n_runs() == 1
        assert tracker.mean_verifier_calls() == 40

    def test_accuracy_above_threshold(self):
        tracker = RegretTracker(100)
        tracker.add(make_result(true_value=0.9))
        tracker.add(make_result(true_value=0.7))
        acc = tracker.mean_accuracy(threshold=0.8)
        assert abs(acc - 0.5) < 1e-9

    def test_multiple_runs_mean(self):
        tracker = RegretTracker(100)
        for calls in [30, 40, 50]:
            tracker.add(make_result(calls=calls))
        assert abs(tracker.mean_verifier_calls() - 40.0) < 1e-9


class TestComputeAccuracy:
    def test_all_above(self):
        results = [make_result(true_value=0.9) for _ in range(5)]
        mean, std = compute_accuracy(results, threshold=0.8)
        assert mean == 1.0
        assert std == 0.0

    def test_half_above(self):
        r1 = [make_result(true_value=0.9) for _ in range(5)]
        r2 = [make_result(true_value=0.5) for _ in range(5)]
        mean, _ = compute_accuracy(r1 + r2, threshold=0.8)
        assert abs(mean - 0.5) < 1e-9


class TestVerifierEfficiency:
    def test_dva_fewer_than_uniform(self):
        dva = [make_result(calls=40) for _ in range(5)]
        uni = [make_result(calls=100) for _ in range(5)]
        ratio, _ = compute_verifier_efficiency(dva, uni)
        assert abs(ratio - 0.4) < 1e-9


class TestEstimateLipschitz:
    def test_exact_lipschitz(self):
        """Trajectories built with L=0.3 should estimate L_hat ≈ 0.15 (uniform mean)."""
        rng = np.random.default_rng(0)
        L = 0.3
        trajectories = []
        for _ in range(200):
            traj = [0.5]
            for _ in range(11):
                delta = rng.uniform(-L, L)
                traj.append(float(np.clip(traj[-1] + delta, 0, 1)))
            trajectories.append(traj)
        L_hat, L_std = estimate_lipschitz(trajectories)
        # Mean of uniform(-L, L) absolute values is L/2
        assert 0.05 <= L_hat <= L + 0.05

    def test_empty(self):
        L_hat, L_std = estimate_lipschitz([])
        assert L_hat == 0.0


class TestFitRegretExponent:
    def test_sqrt_rate(self):
        budgets = [100, 200, 400, 800, 1600]
        regrets = [2.5 * np.sqrt(b) for b in budgets]
        exponent, _, r2 = fit_regret_exponent(budgets, regrets)
        assert abs(exponent - 0.5) < 0.01
        assert r2 > 0.999

    def test_linear_rate(self):
        budgets = [100, 200, 400, 800]
        regrets = [0.01 * b for b in budgets]
        exponent, _, r2 = fit_regret_exponent(budgets, regrets)
        assert abs(exponent - 1.0) < 0.02
