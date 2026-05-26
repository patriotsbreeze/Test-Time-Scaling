"""
DVA-MCTS: Dynamic Verifier Allocation Monte Carlo Tree Search.

Core algorithm implementing Theorem 1 (regret bound) and Theorem 2
(sub-linear verifier calls) from the paper.

Key design decision — dynamic Lipschitz proxy:
    Rather than caching a proxy_value once, the algorithm recomputes the
    nearest-verified-ancestor estimate fresh at each step via
    node.get_dynamic_estimate(L).  This ensures the estimation error at
    step t is bounded by L·Δd_t(s) — where Δd_t decreases as ancestors
    are verified over time — without requiring additional verifier calls.
    Proof correctness depends on this dynamic recomputation (Appendix A.1).
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
    adaptive_l_estimate: float = 0.0  # running-max L̂ (0 when not using adaptive)


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

    def is_accurate(self, threshold: float = 0.8) -> bool:
        """Return True if the best solution's true quality meets *threshold*."""
        return self.best_true_value >= threshold


class DVAMCTS:
    """
    Dynamic Verifier Allocation Monte Carlo Tree Search.

    At each search step the algorithm decides whether to invoke the verifier
    based on a budget-sensitive threshold:

        tau_t = gamma / t^alpha

    The verifier is called iff the Lipschitz-estimated score discrepancy
    between the current node and its nearest verified ancestor exceeds tau_t.
    When skipped, the dynamic Lipschitz proxy provides a fresh estimate using
    the nearest currently-verified ancestor's score (the Lipschitz midpoint).

    This achieves O(sqrt(T log K)) regret against the oracle while making
    at most N_T = K(K^D - 1)/(K - 1) total verifier calls (Theorems 1 & 2).

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

        # Reset adaptive L̂ for a fresh run
        self._adaptive_l = self.cfg.adaptive_l_init

        # Step 0: always verify the root to warm-start (gives anchor for proxy)
        if self.cfg.always_verify_root:
            self._call_verifier(tree.root)
            verifier_calls_total += 1

        for t in range(1, budget + 1):
            # ── Selection (dynamic UCT with Lipschitz proxy) ────────────────
            node = self._select(tree)

            # ── Expansion: add children if at unexpanded leaf ───────────────
            if not tree.at_max_depth(node) and not node.children:
                tree.expand(node)

            # Pick the single simulation target
            if node.children:
                L_current = self._current_L()
                sim_node = max(
                    node.children,
                    key=lambda c: c.uct_score_dynamic(node.visit_count, self.cfg.c_ucb, L_current),
                )
            else:
                sim_node = node  # at max depth — score current leaf

            # ── Verifier decision for the simulation node ──────────────────
            tau = self.cfg.threshold(t)
            if self._should_call_verifier(sim_node, tau):
                self._call_verifier(sim_node)
                verifier_calls_total += 1
                # Assertion: single-verification property
                assert sim_node.verifier_called, "verifier_called must be set after _call_verifier"
                if self.cfg.use_adaptive_l:
                    self._update_adaptive_l(sim_node)
            # No else: proxy is computed dynamically at each evaluation,
            # no persistent proxy_value stored (see module docstring).

            # ── Backpropagation with dynamic proxy estimate ─────────────────
            L_current = self._current_L()
            sim_value = sim_node.get_dynamic_estimate(L_current)
            if sim_value is None:
                sim_value = 0.5  # uninformative prior if no ancestor exists
            self._backpropagate(sim_node, sim_value)

            # ── Record step ────────────────────────────────────────────────
            _, oracle_best = tree.best_true_leaf()

            all_visited = [n for n in tree._node_registry.values() if n.visit_count > 0]
            if all_visited:
                alg_best_node = max(
                    all_visited,
                    key=lambda n: (
                        n.get_dynamic_estimate(L_current)
                        if n.get_dynamic_estimate(L_current) is not None
                        else -1.0
                    ),
                )
                alg_true = alg_best_node.true_value or 0.0
            else:
                alg_true = 0.0

            cumulative_regret += max(0.0, oracle_best - alg_true)

            dyn_est = sim_node.get_dynamic_estimate(L_current)
            rec = StepRecord(
                step=t,
                node_id=sim_node.node_id,
                depth=sim_node.depth,
                verifier_called=sim_node.verifier_called,
                estimated_value=dyn_est if dyn_est is not None else 0.0,
                true_value=sim_node.true_value or 0.0,
                oracle_best_true=oracle_best,
                cumulative_regret=cumulative_regret,
                total_verifier_calls=verifier_calls_total,
                adaptive_l_estimate=self._adaptive_l if self.cfg.use_adaptive_l else 0.0,
            )
            step_records.append(rec)

        # ── Final answer ───────────────────────────────────────────────────
        L_current = self._current_L()
        best_node, best_est = tree.best_estimated_leaf(L=L_current)
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

    def _current_L(self) -> float:
        """Return the current Lipschitz constant (adaptive or fixed)."""
        return self._adaptive_l if self.cfg.use_adaptive_l else self.cfg.lipschitz_L

    def _select(self, tree: SearchTree) -> SearchNode:
        """UCT-based tree traversal using dynamic Lipschitz proxy estimates.

        Descends from root, always selecting the child with the highest
        dynamic UCT score.  Stops at the first unexpanded node (no children
        yet) or at max-depth leaves — whichever comes first.
        """
        node = tree.root
        L = self._current_L()

        while node.children:
            # Has children: select best by dynamic UCT
            node = max(
                node.children,
                key=lambda c: c.uct_score_dynamic(node.visit_count, self.cfg.c_ucb, L),
            )
            # Stop if we've reached max depth
            if tree.at_max_depth(node):
                break

        return node

    def _should_call_verifier(self, node: SearchNode, tau: float) -> bool:
        """
        Invoke the verifier iff the Lipschitz-estimated discrepancy to the
        nearest verified ancestor exceeds the current threshold tau_t.

            delta_t = L * |d(node) - d(ancestor)|
            b_t = 1{not yet verified} AND 1{delta_t > tau_t OR no ancestor}

        Once a node is verified (verifier_called=True), it is NEVER
        re-verified (single-verification property, Lemma A.3).
        """
        if node.verifier_called:
            return False  # single-verification ceiling: never re-verify

        ancestor = node.nearest_verified_ancestor()
        if ancestor is None:
            return True  # no anchor: must call to initialise proxy chain

        depth_gap = node.depth - ancestor.depth
        L = self._current_L()
        delta = L * depth_gap
        return delta > tau

    def _update_adaptive_l(self, node: SearchNode) -> None:
        """Update running-max L̂ from a newly verified node.

        Compares verifier_value at this node against all verified ancestors,
        taking the running maximum of |score_gap| / depth_gap over all pairs.
        Converges to a conservative over-estimate of the true Lipschitz
        constant L (empirically: L̂ ≈ 1.23L, Section 5.4).
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
        """Invoke the verifier and store the noisy score on the node.

        This is the ONLY method that sets node.verifier_called = True.
        After this call, node.verifier_value holds the noisy observation
        V(s) + epsilon, and node.get_dynamic_estimate(L) returns this value
        for all future steps — no re-verification needed.
        """
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        """Propagate the simulation value back to the root."""
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent
