"""
DVA-MCTS: Dynamic Verifier Allocation Monte Carlo Tree Search.

Core algorithm implementing Theorem 1 (regret bound) and Theorem 2
(sub-linear verifier calls) from the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import DVAConfig
from .tree import SearchNode, SearchTree
from .verifier import Verifier


@dataclass
class StepRecord:
    """Records the state of the algorithm at a single search step."""

    step: int
    node_id: int
    depth: int
    verifier_called: bool
    estimated_value: float
    true_value: float
    oracle_best_true: float
    cumulative_regret: float
    total_verifier_calls: int
    adaptive_l_estimate: float = 0.0  # running max L estimate (0 when not using adaptive)


@dataclass
class SearchResult:
    """Complete result from a DVA-MCTS run."""

    best_node_id: int
    best_estimated_value: float
    best_true_value: float
    oracle_best_true_value: float

    total_steps: int
    total_verifier_calls: int
    verifier_call_fraction: float

    # Per-step records for analysis
    step_records: List[StepRecord] = field(default_factory=list)

    @property
    def final_regret(self) -> float:
        if not self.step_records:
            return self.oracle_best_true_value - self.best_true_value
        return self.step_records[-1].cumulative_regret

    @property
    def accuracy(self, threshold: float = 0.8) -> bool:
        return self.best_true_value >= threshold


class DVAMCTS:
    """
    Dynamic Verifier Allocation Monte Carlo Tree Search.

    At each search step the algorithm decides whether to invoke the verifier
    based on a budget-sensitive threshold:

        tau_t = gamma / t^alpha

    The verifier is called iff the Lipschitz-estimated score discrepancy
    between the current node and its nearest verified ancestor exceeds tau_t.
    When skipped, the Lipschitz proxy provides a conservative estimate.

    This achieves O(sqrt(T log K)) regret against the oracle while making
    only O(sqrt(T) log T) verifier calls (Theorems 1 & 2).

    Parameters
    ----------
    verifier : Verifier
        The process reward model (or any Verifier implementation).
    config : DVAConfig
        Hyperparameters for the algorithm.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        verifier: Verifier,
        config: DVAConfig,
        rng: np.random.Generator,
    ) -> None:
        self.verifier = verifier
        self.cfg = config
        self.rng = rng
        self._adaptive_l = config.adaptive_l_init

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        """
        Run DVA-MCTS for `budget` steps on the given tree.

        Returns a SearchResult with per-step records for full analysis.
        """
        step_records: List[StepRecord] = []
        verifier_calls_total = 0
        cumulative_regret = 0.0

        # Reset adaptive L for a fresh run
        self._adaptive_l = self.cfg.adaptive_l_init

        # Step 0: always verify the root to warm-start
        if self.cfg.always_verify_root:
            self._call_verifier(tree.root)
            verifier_calls_total += 1

        for t in range(1, budget + 1):
            # ── Selection ──────────────────────────────────────────────────
            node = self._select(tree)

            # ── Expansion: add children to tree but only simulate one ──────
            if tree.is_leaf(node):
                tree.expand(node)

            # Pick the single simulation target
            if node.children:
                # Select child with highest UCT (may be inf for unvisited)
                sim_node = max(
                    node.children,
                    key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb),
                )
            else:
                sim_node = node  # at max depth, score current leaf

            # ── Verifier decision for the simulation node ──────────────────
            tau = self.cfg.threshold(t)
            if self._should_call_verifier(sim_node, tau):
                self._call_verifier(sim_node)
                verifier_calls_total += 1
                if self.cfg.use_adaptive_l:
                    self._update_adaptive_l(sim_node)
            else:
                self._apply_proxy(sim_node)

            # ── Backpropagation ────────────────────────────────────────────
            sim_value = sim_node.estimated_value or 0.0
            self._backpropagate(sim_node, sim_value)

            # ── Record step ────────────────────────────────────────────────
            _, oracle_best = tree.best_true_leaf()

            all_visited = [n for n in tree._node_registry.values() if n.visit_count > 0]
            if all_visited:
                alg_best_node = max(
                    all_visited,
                    key=lambda n: n.estimated_value if n.estimated_value is not None else -1.0,
                )
                alg_true = alg_best_node.true_value or 0.0
            else:
                alg_true = 0.0

            cumulative_regret += max(0.0, oracle_best - alg_true)

            rec = StepRecord(
                step=t,
                node_id=sim_node.node_id,
                depth=sim_node.depth,
                verifier_called=sim_node.verifier_called,
                estimated_value=sim_node.estimated_value or 0.0,
                true_value=sim_node.true_value or 0.0,
                oracle_best_true=oracle_best,
                cumulative_regret=cumulative_regret,
                total_verifier_calls=verifier_calls_total,
                adaptive_l_estimate=self._adaptive_l if self.cfg.use_adaptive_l else 0.0,
            )
            step_records.append(rec)

        # ── Final answer ───────────────────────────────────────────────────
        best_node, best_est = tree.best_estimated_leaf()
        _, oracle_best = tree.best_true_leaf()
        best_true = best_node.true_value or 0.0

        return SearchResult(
            best_node_id=best_node.node_id,
            best_estimated_value=best_est,
            best_true_value=best_true,
            oracle_best_true_value=oracle_best,
            total_steps=budget,
            total_verifier_calls=verifier_calls_total,
            verifier_call_fraction=verifier_calls_total / max(budget, 1),
            step_records=step_records,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _select(self, tree: SearchTree) -> SearchNode:
        """UCT-based tree traversal: descend until a leaf is reached."""
        node = tree.root
        while not tree.is_leaf(node):
            if not node.children:
                break
            # Select child maximising UCT score
            node = max(
                node.children,
                key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb),
            )
        return node

    def _should_call_verifier(self, node: SearchNode, tau: float) -> bool:
        """
        Invoke the verifier iff: node is unverified AND Lipschitz-estimated
        discrepancy from nearest verified ancestor exceeds tau.

        delta = L * |d(node) - d(ancestor)|
        b_t   = 1{not verified} AND 1{delta > tau}
        """
        if node.verifier_called:
            return False

        ancestor = node.nearest_verified_ancestor()
        if ancestor is None:
            return True

        depth_gap = abs(node.depth - ancestor.depth)
        L = self._adaptive_l if self.cfg.use_adaptive_l else self.cfg.lipschitz_L
        delta = L * depth_gap
        return delta > tau

    def _update_adaptive_l(self, node: SearchNode) -> None:
        """Update running-max L estimate from a newly verified node.

        Compares verifier_value at this node against all verified ancestors,
        taking the max of |score_gap| / depth_gap over all pairs.
        """
        if node.verifier_value is None:
            return
        current: Optional[SearchNode] = node.parent
        while current is not None:
            if current.verifier_called and current.verifier_value is not None:
                depth_gap = node.depth - current.depth
                if depth_gap > 0:
                    ratio = abs(node.verifier_value - current.verifier_value) / depth_gap
                    if ratio > self._adaptive_l:
                        self._adaptive_l = ratio
            current = current.parent

    def _call_verifier(self, node: SearchNode) -> None:
        """Invoke the verifier and store the result on the node."""
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score
        # Sync true_value from the verifier's internal process (for regret tracking)
        if hasattr(self.verifier, "true_value"):
            node.true_value = self.verifier.true_value(node)

    def _apply_proxy(self, node: SearchNode) -> None:
        """
        Compute a Lipschitz proxy for a node without calling the verifier.

        proxy = V_hat(ancestor)
        Guaranteed error: |proxy - V(node)| <= L * depth_gap + sigma
        """
        # Sync true_value even when skipping verifier (for regret tracking)
        if hasattr(self.verifier, "true_value"):
            node.true_value = self.verifier.true_value(node)

        ancestor = node.nearest_verified_ancestor()
        if ancestor is None or ancestor.verifier_value is None:
            if node.parent and node.parent.estimated_value is not None:
                node.proxy_value = node.parent.estimated_value
            else:
                node.proxy_value = 0.5  # neutral prior
            return

        node.proxy_value = ancestor.verifier_value

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        """Propagate the simulation value back to the root."""
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent
