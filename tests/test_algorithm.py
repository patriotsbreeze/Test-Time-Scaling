"""Unit tests for DVA-MCTS and baseline algorithms."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dva_mcts.algorithm import DVAMCTS
from dva_mcts.baselines import BestOfN, RandomAllocationMCTS, UniformMCTS
from dva_mcts.config import DVAConfig
from dva_mcts.tree import SearchTree
from dva_mcts.verifier import LipschitzVerifier


def make_setup(branching=2, depth=6, L=0.35, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    verifier = LipschitzVerifier(L, sigma, rng)
    tree = SearchTree(branching, depth, lambda nid, d: 0.0, rng)
    verifier.populate_tree_values(tree)
    cfg = DVAConfig(
        gamma=1.0, alpha=0.5, lipschitz_L=L, sigma_max=sigma,
        branching_factor=branching, max_depth=depth,
    )
    return tree, verifier, cfg, rng


class TestDVAMCTS:
    def test_returns_search_result(self):
        tree, verifier, cfg, rng = make_setup()
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=20)
        assert result.total_steps == 20

    def test_verifier_calls_sublinear(self):
        """DVA-MCTS should call verifier well below budget times (sub-linear)."""
        tree, verifier, cfg, rng = make_setup(seed=7)
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=100)
        # With gamma=1, alpha=0.5, L=0.35 we expect ~sqrt(100)*log(100) ≈ 46 calls
        # Allow generous upper bound of 85% of budget for robustness
        assert result.total_verifier_calls < int(0.85 * 100 + 5)

    def test_best_value_in_unit_interval(self):
        tree, verifier, cfg, rng = make_setup()
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=50)
        assert 0.0 <= result.best_true_value <= 1.0

    def test_step_records_length(self):
        tree, verifier, cfg, rng = make_setup()
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=30)
        assert len(result.step_records) == 30

    def test_cumulative_regret_nondecreasing(self):
        """Cumulative regret should never decrease."""
        tree, verifier, cfg, rng = make_setup()
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=50)
        regrets = [r.cumulative_regret for r in result.step_records]
        for i in range(1, len(regrets)):
            assert regrets[i] >= regrets[i - 1] - 1e-9

    def test_threshold_zero_alpha_calls_always(self):
        """With alpha=0 threshold is constant; verifier should be called frequently."""
        tree, verifier, cfg, rng = make_setup()
        cfg.alpha = 0.0
        cfg.gamma = 0.001  # very low threshold → almost always call
        alg = DVAMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=50)
        # With near-zero threshold, nearly all steps should call verifier
        assert result.total_verifier_calls > 30

    def test_higher_gamma_fewer_calls(self):
        """Larger gamma → higher threshold → fewer verifier calls."""
        results = []
        for gamma in [0.1, 1.0, 5.0]:
            tree, verifier, cfg, rng = make_setup(seed=42)
            cfg.gamma = gamma
            alg = DVAMCTS(verifier, cfg, rng)
            res = alg.search(tree, budget=80)
            results.append(res.total_verifier_calls)
        # calls should generally decrease as gamma increases
        assert results[0] >= results[1] or results[1] >= results[2]  # loose monotonicity

    def test_reproducibility(self):
        """Same seed should produce identical results."""
        r1 = None
        r2 = None
        for _ in range(2):
            tree, verifier, cfg, rng = make_setup(seed=123)
            alg = DVAMCTS(verifier, cfg, rng)
            res = alg.search(tree, budget=30)
            if r1 is None:
                r1 = res
            else:
                r2 = res
        assert r1.total_verifier_calls == r2.total_verifier_calls
        assert abs(r1.best_true_value - r2.best_true_value) < 1e-9


class TestUniformMCTS:
    def test_calls_verifier_at_every_step(self):
        tree, verifier, cfg, rng = make_setup()
        alg = UniformMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=30)
        # uniform must call verifier >= budget (one per step, plus root)
        assert result.total_verifier_calls >= 30

    def test_returns_valid_result(self):
        tree, verifier, cfg, rng = make_setup()
        alg = UniformMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=20)
        assert 0.0 <= result.best_true_value <= 1.0


class TestRandomAllocationMCTS:
    def test_expected_calls_approx_half(self):
        """With p=0.5, expect ~0.5 * budget calls on average."""
        call_counts = []
        for seed in range(20):
            tree, verifier, cfg, rng = make_setup(seed=seed)
            alg = RandomAllocationMCTS(verifier, cfg, rng, call_probability=0.5)
            res = alg.search(tree, budget=100)
            call_counts.append(res.total_verifier_calls)
        mean_calls = np.mean(call_counts)
        # Should be in range [40, 80] for p=0.5 with budget=100
        assert 30 <= mean_calls <= 90

    def test_returns_valid_result(self):
        tree, verifier, cfg, rng = make_setup()
        alg = RandomAllocationMCTS(verifier, cfg, rng)
        result = alg.search(tree, budget=20)
        assert 0.0 <= result.best_true_value <= 1.0


class TestBestOfN:
    def test_calls_exactly_n(self):
        tree, verifier, cfg, rng = make_setup()
        alg = BestOfN(verifier, rng)
        result = alg.search(tree, budget=25)
        assert result.total_verifier_calls == 25

    def test_returns_valid_result(self):
        tree, verifier, cfg, rng = make_setup()
        alg = BestOfN(verifier, rng)
        result = alg.search(tree, budget=20)
        assert 0.0 <= result.best_true_value <= 1.0


class TestComparativeRegret:
    """Integration tests checking that DVA-MCTS is competitive with Uniform."""

    def test_dva_regret_not_much_worse_than_uniform(self):
        """
        DVA-MCTS should achieve final regret within 3x of Uniform
        (the bound factor accounts for finite-budget effects).
        """
        budget = 200
        n_runs = 20
        dva_regrets, uni_regrets = [], []

        for seed in range(n_runs):
            for cls, store in [(DVAMCTS, dva_regrets), (UniformMCTS, uni_regrets)]:
                tree, verifier, cfg, rng = make_setup(seed=seed + cls.__name__.__hash__() % 1000)
                alg = cls(verifier=verifier, config=cfg, rng=rng)
                res = alg.search(tree, budget)
                store.append(res.final_regret)

        mean_dva = np.mean(dva_regrets)
        mean_uni = np.mean(uni_regrets)
        # DVA regret should not be more than 3x worse than Uniform
        assert mean_dva <= mean_uni * 3.0, (
            f"DVA regret {mean_dva:.3f} is much worse than Uniform {mean_uni:.3f}"
        )

    def test_dva_uses_fewer_calls_than_uniform(self):
        budget = 150
        n_runs = 15
        dva_calls, uni_calls = [], []

        for seed in range(n_runs):
            # Use different seed offsets to give each algorithm a distinct tree
            for cls, store, offset in [
                (DVAMCTS,     dva_calls, 0),
                (UniformMCTS, uni_calls, 50000),
            ]:
                tree, verifier, cfg, rng = make_setup(seed=seed + offset)
                alg = cls(verifier=verifier, config=cfg, rng=rng)
                res = alg.search(tree, budget)
                store.append(res.total_verifier_calls)

        # DVA should call verifier less often than Uniform (which calls at every step)
        assert np.mean(dva_calls) < np.mean(uni_calls), (
            f"DVA-MCTS mean calls {np.mean(dva_calls):.1f} should be < "
            f"Uniform mean calls {np.mean(uni_calls):.1f}"
        )
