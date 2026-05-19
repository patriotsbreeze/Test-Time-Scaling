"""
Verifier interfaces and implementations.

A Verifier maps a SearchNode to a score in [0, 1].  The base class defines
the interface; concrete implementations include:

  - LipschitzVerifier: synthetic Lipschitz-L score process + Gaussian noise.
    Calibrated to match reported PRM score distributions (Lightman et al. 2023).
  - OracleVerifier:    returns the true value without noise (evaluation only).
  - CachedVerifier:    wraps any verifier and caches calls to avoid redundant work.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np

from .tree import SearchNode


class Verifier(ABC):
    """Abstract base class for all verifiers."""

    @abstractmethod
    def score(self, node: SearchNode) -> float:
        """Return a score in [0, 1] for the given node."""

    def __call__(self, node: SearchNode) -> float:
        return self.score(node)

    @property
    def call_count(self) -> int:
        return self._call_count

    def reset_count(self) -> None:
        self._call_count = 0

    _call_count: int = 0


class LipschitzVerifier(Verifier):
    """
    Synthetic verifier that obeys the Lipschitz continuity assumption.

    The true value function is constructed as a Lipschitz-L process:
        V(node) = clip(V(parent) + Delta, 0, 1)
    where Delta ~ Uniform(-L, L).

    Observed scores include additive sub-Gaussian noise:
        hat_V(node) = V(node) + epsilon,   epsilon ~ N(0, sigma^2)

    Parameters
    ----------
    lipschitz_L : float
        Lipschitz constant.  Controls how fast scores can change with depth.
    sigma : float
        Std dev of Gaussian observation noise.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    def __init__(self, lipschitz_L: float, sigma: float, rng: np.random.Generator) -> None:
        self.L = lipschitz_L
        self.sigma = sigma
        self.rng = rng
        self._true_values: Dict[int, float] = {}
        self._call_count = 0

    def _get_true_value(self, node: SearchNode) -> float:
        """Compute and cache the true (noiseless) value for a node."""
        if node.node_id in self._true_values:
            return self._true_values[node.node_id]

        if node.parent is None:
            # Root: start from a neutral prior
            v = self.rng.uniform(0.3, 0.7)
        else:
            parent_v = self._get_true_value(node.parent)
            delta = self.rng.uniform(-self.L, self.L)
            v = float(np.clip(parent_v + delta, 0.0, 1.0))

        self._true_values[node.node_id] = v
        return v

    def true_value(self, node: SearchNode) -> float:
        """Return the true (noiseless) value — used for evaluation only."""
        return self._get_true_value(node)

    def score(self, node: SearchNode) -> float:
        """Return a noisy observation of the node's true value."""
        self._call_count += 1
        true_v = self._get_true_value(node)
        noise = self.rng.normal(0.0, self.sigma)
        return float(np.clip(true_v + noise, 0.0, 1.0))

    def populate_tree_values(self, tree) -> None:
        """Pre-populate true_value fields on all registered nodes in a SearchTree."""
        for node in tree._node_registry.values():
            node.true_value = self._get_true_value(node)


class OracleVerifier(Verifier):
    """Returns the true value without noise. Only used for evaluation."""

    def __init__(self, base_verifier: LipschitzVerifier) -> None:
        self._base = base_verifier
        self._call_count = 0

    def score(self, node: SearchNode) -> float:
        self._call_count += 1
        return self._base.true_value(node)


class CachedVerifier(Verifier):
    """Wraps a verifier and caches results so re-scoring the same node is free."""

    def __init__(self, base: Verifier) -> None:
        self._base = base
        self._cache: Dict[int, float] = {}
        self._call_count = 0

    def score(self, node: SearchNode) -> float:
        if node.node_id not in self._cache:
            self._cache[node.node_id] = self._base.score(node)
            self._call_count += 1
        return self._cache[node.node_id]

    def evict(self, node_id: int) -> None:
        self._cache.pop(node_id, None)
