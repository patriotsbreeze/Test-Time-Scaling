"""Unit tests for Verifier implementations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dva_mcts.tree import SearchNode
from dva_mcts.verifier import CachedVerifier, LipschitzVerifier, OracleVerifier


def make_verifier(L=0.35, sigma=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return LipschitzVerifier(lipschitz_L=L, sigma=sigma, rng=rng)


def make_chain(depth=5):
    """Return a root-to-leaf chain of SearchNodes."""
    root = SearchNode(node_id=0, depth=0)
    nodes = [root]
    for i in range(1, depth + 1):
        n = SearchNode(node_id=i, depth=i, parent=nodes[-1])
        nodes.append(n)
    return nodes


class TestLipschitzVerifier:
    def test_score_in_unit_interval(self):
        v = make_verifier()
        for node in make_chain(8):
            s = v.score(node)
            assert 0.0 <= s <= 1.0

    def test_true_value_lipschitz(self):
        """Verify |V(child) - V(parent)| <= L for all adjacent pairs."""
        L = 0.35
        v = make_verifier(L=L)
        nodes = make_chain(20)
        for i in range(1, len(nodes)):
            vp = v.true_value(nodes[i - 1])
            vc = v.true_value(nodes[i])
            assert abs(vc - vp) <= L + 1e-9, f"Lipschitz violated at depth {i}: |{vc}-{vp}|={abs(vc-vp)}"

    def test_true_value_cached(self):
        """Calling true_value twice should return the same value."""
        v = make_verifier()
        nodes = make_chain(3)
        v1 = v.true_value(nodes[2])
        v2 = v.true_value(nodes[2])
        assert v1 == v2

    def test_call_count(self):
        v = make_verifier()
        assert v.call_count == 0
        nodes = make_chain(3)
        for n in nodes:
            v.score(n)
        assert v.call_count == len(nodes)

    def test_score_noise(self):
        """Noisy score should differ from true value but be close (sigma=0.05)."""
        v = make_verifier(sigma=0.05, seed=99)
        nodes = make_chain(5)
        errors = []
        for n in nodes:
            true = v.true_value(n)
            observed = v.score(n)
            errors.append(abs(observed - true))
        # Mean error should be close to sigma * sqrt(2/pi) ≈ 0.04 for half-normal
        assert np.mean(errors) < 0.15, "Noise too large"

    def test_zero_noise_oracle(self):
        v = make_verifier(sigma=0.0)
        nodes = make_chain(4)
        for n in nodes:
            assert abs(v.score(n) - v.true_value(n)) < 1e-12


class TestOracleVerifier:
    def test_noiseless(self):
        base = make_verifier(sigma=0.1)
        oracle = OracleVerifier(base)
        nodes = make_chain(5)
        for n in nodes:
            expected = base.true_value(n)
            assert oracle.score(n) == expected


class TestCachedVerifier:
    def test_caches_calls(self):
        base = make_verifier()
        cached = CachedVerifier(base)
        node = make_chain(1)[0]
        s1 = cached.score(node)
        s2 = cached.score(node)
        assert s1 == s2
        assert cached.call_count == 1  # second call was cached
        assert base.call_count == 1

    def test_evict_forces_recompute(self):
        base = make_verifier(sigma=0.0)
        cached = CachedVerifier(base)
        node = make_chain(1)[0]
        cached.score(node)
        cached.evict(node.node_id)
        cached.score(node)
        assert cached.call_count == 2
