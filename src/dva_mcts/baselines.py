"""
Baseline search algorithms for comparison with DVA-MCTS.

  - UniformMCTS:       verifier called at every node (exhaustive verification)
  - RandomAllocationMCTS: verifier called with fixed probability p=0.5
  - RandomPathSearch:  N independent uniform-random root-to-leaf paths,
                       each fully verified; demonstrates that exploration
                       strategy (not just verification) matters in deep trees.
                       NOTE: this is NOT standard Best-of-N sampling (which
                       uses the LLM's own generation distribution). It is an
                       unguided random walk included only to show that
                       *guided* tree search is necessary in high-branching
                       trees (Section 6.1 of the paper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .algorithm import SearchResult, StepRecord
from .config import DVAConfig
from .tree import SearchNode, SearchTree
from .verifier import Verifier


class UniformMCTS:
    """
    Standard MCTS with verifier called at every expanded node.

    This is the 'exhaustive verification' baseline against which DVA-MCTS
    is measured. Uses UCT selection identical to DVA-MCTS (dynamic estimates
    are equivalent to verifier estimates for Uniform since every node is
    verified).
    """

    def __init__(self, verifier: Verifier, config: DVAConfig, rng: np.random.Generator) -> None:
        self.verifier = verifier
        self.cfg = config
        self.rng = rng

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        step_records: List[StepRecord] = []
        verifier_calls = 0
        cumulative_regret = 0.0

        # Verify root for warm-start
        self._call_verifier(tree.root)
        verifier_calls += 1

        for t in range(1, budget + 1):
            node = self._select(tree)
            if not tree.at_max_depth(node) and not node.children:
                tree.expand(node)

            if node.children:
                sim_node = max(
                    node.children,
                    key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb),
                )
            else:
                sim_node = node

            self._call_verifier(sim_node)
            verifier_calls += 1

            sim_value = sim_node.estimated_value or 0.0
            self._backpropagate(sim_node, sim_value)

            _, oracle_best = tree.best_true_leaf()
            all_visited = [n for n in tree._node_registry.values() if n.visit_count > 0]
            if all_visited:
                alg_best = max(
                    all_visited,
                    key=lambda n: n.estimated_value if n.estimated_value is not None else -1.0,
                )
                alg_true = alg_best.true_value or 0.0
            else:
                alg_true = 0.0

            cumulative_regret += max(0.0, oracle_best - alg_true)
            step_records.append(StepRecord(
                step=t,
                node_id=sim_node.node_id,
                depth=sim_node.depth,
                verifier_called=True,
                estimated_value=sim_node.estimated_value or 0.0,
                true_value=sim_node.true_value or 0.0,
                oracle_best_true=oracle_best,
                cumulative_regret=cumulative_regret,
                total_verifier_calls=verifier_calls,
            ))

        best_node, best_est = tree.best_estimated_leaf()
        _, oracle_best = tree.best_true_leaf()

        return SearchResult(
            best_node_id=best_node.node_id,
            best_estimated_value=best_est,
            best_true_value=best_node.true_value or 0.0,
            oracle_best_true_value=oracle_best,
            total_steps=budget,
            total_verifier_calls=verifier_calls,
            verifier_call_fraction=verifier_calls / max(budget, 1),
            step_records=step_records,
        )

    def _select(self, tree: SearchTree) -> SearchNode:
        node = tree.root
        while node.children and not tree.at_max_depth(node):
            node = max(node.children, key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
        return node

    def _call_verifier(self, node: SearchNode) -> None:
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent


class RandomAllocationMCTS:
    """
    MCTS where each node is verified independently with probability p.
    An uninformed baseline that achieves sub-linear calls but carries no
    regret guarantee.
    """

    def __init__(
        self,
        verifier: Verifier,
        config: DVAConfig,
        rng: np.random.Generator,
        call_probability: float = 0.5,
    ) -> None:
        self.verifier = verifier
        self.cfg = config
        self.rng = rng
        self.p = call_probability

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        step_records: List[StepRecord] = []
        verifier_calls = 0
        cumulative_regret = 0.0

        self._call_verifier(tree.root)
        verifier_calls += 1

        for t in range(1, budget + 1):
            node = self._select(tree)
            if not tree.at_max_depth(node) and not node.children:
                tree.expand(node)

            if node.children:
                sim_node = max(
                    node.children,
                    key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb),
                )
            else:
                sim_node = node

            if self.rng.random() < self.p:
                self._call_verifier(sim_node)
                verifier_calls += 1
            # Else: use parent's estimated value as proxy (UCT will use this
            # via the inherited mean_value; no persistent proxy stored).

            sim_value = sim_node.estimated_value or (
                sim_node.parent.estimated_value if sim_node.parent else 0.5
            ) or 0.5
            self._backpropagate(sim_node, sim_value)

            _, oracle_best = tree.best_true_leaf()
            all_visited = [n for n in tree._node_registry.values() if n.visit_count > 0]
            if all_visited:
                alg_best = max(
                    all_visited,
                    key=lambda n: n.estimated_value if n.estimated_value is not None else -1.0,
                )
                alg_true = alg_best.true_value or 0.0
            else:
                alg_true = 0.0

            cumulative_regret += max(0.0, oracle_best - alg_true)
            step_records.append(StepRecord(
                step=t,
                node_id=sim_node.node_id,
                depth=sim_node.depth,
                verifier_called=sim_node.verifier_called,
                estimated_value=sim_node.estimated_value or 0.0,
                true_value=sim_node.true_value or 0.0,
                oracle_best_true=oracle_best,
                cumulative_regret=cumulative_regret,
                total_verifier_calls=verifier_calls,
            ))

        best_node, best_est = tree.best_estimated_leaf()
        _, oracle_best = tree.best_true_leaf()

        return SearchResult(
            best_node_id=best_node.node_id,
            best_estimated_value=best_est,
            best_true_value=best_node.true_value or 0.0,
            oracle_best_true_value=oracle_best,
            total_steps=budget,
            total_verifier_calls=verifier_calls,
            verifier_call_fraction=verifier_calls / max(budget, 1),
            step_records=step_records,
        )

    def _select(self, tree: SearchTree) -> SearchNode:
        node = tree.root
        while node.children and not tree.at_max_depth(node):
            node = max(node.children, key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
        return node

    def _call_verifier(self, node: SearchNode) -> None:
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent


# ── Keep BestOfN name for backward-compat; paper uses "Random-Path Search" ──

class RandomPathSearch:
    """
    Random-Path Search (RPS): sample `budget` independent uniform-random
    root-to-leaf paths; verify each leaf and return the highest-scoring one.

    This is NOT standard Best-of-N sampling (which samples from the LLM's
    own policy distribution).  RPS samples paths uniformly at random from
    the tree, achieving Pr[optimal leaf] = K^{-D}.  For deep trees this
    probability is negligible (2^{-12} ≈ 0.02% for K=2, D=12), so RPS
    serves only to demonstrate that *guided* search is necessary—it is
    not a fair competitor to tree-based methods (Section 6.1).
    """

    def __init__(self, verifier: Verifier, rng: np.random.Generator) -> None:
        self.verifier = verifier
        self.rng = rng

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        verifier_calls = 0
        best_score = -1.0
        best_node: Optional[SearchNode] = None
        running_best_true = -1.0
        cumulative_regret = 0.0
        step_records: List[StepRecord] = []

        for t in range(1, budget + 1):
            # Simulate an independent uniform-random path from root to a leaf
            node = tree.root
            while not tree.at_max_depth(node):
                if not node.children:
                    tree.expand(node)
                if node.children:
                    node = self.rng.choice(node.children)  # type: ignore[arg-type]
                else:
                    break

            score = self.verifier.score(node)
            verifier_calls += 1
            node.verifier_called = True
            node.verifier_value = score
            node.visit_count += 1
            node.total_value += score

            if score > best_score:
                best_score = score
                best_node = node

            true_val = node.true_value or 0.0
            if true_val > running_best_true:
                running_best_true = true_val

            _, oracle_best = tree.best_true_leaf()
            cumulative_regret += max(0.0, oracle_best - running_best_true)

            step_records.append(StepRecord(
                step=t,
                node_id=node.node_id,
                depth=node.depth,
                verifier_called=True,
                estimated_value=score,
                true_value=true_val,
                oracle_best_true=oracle_best,
                cumulative_regret=cumulative_regret,
                total_verifier_calls=verifier_calls,
            ))

        if best_node is None:
            best_node = tree.root

        _, oracle_best = tree.best_true_leaf()
        best_true = best_node.true_value or 0.0

        return SearchResult(
            best_node_id=best_node.node_id,
            best_estimated_value=best_score,
            best_true_value=best_true,
            oracle_best_true_value=oracle_best,
            total_steps=budget,
            total_verifier_calls=verifier_calls,
            verifier_call_fraction=1.0,
            step_records=step_records,
        )


# Backward-compatibility alias
BestOfN = RandomPathSearch
