"""
Baseline search algorithms for comparison with DVA-MCTS.

  - UniformMCTS:          verifier called at every node (exhaustive verification)
  - RandomAllocationMCTS: verifier called with fixed probability p=0.5
  - BestOfN:              N independent samples, each fully verified; no tree search
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
    is measured. Uses UCT selection identical to DVA-MCTS.
    """

    def __init__(self, verifier: Verifier, config: DVAConfig, rng: np.random.Generator) -> None:
        self.verifier = verifier
        self.cfg = config
        self.rng = rng

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        step_records: List[StepRecord] = []
        verifier_calls = 0
        cumulative_regret = 0.0

        # Verify root
        self._call_verifier(tree.root)
        verifier_calls += 1

        for t in range(1, budget + 1):
            node = self._select(tree)
            if tree.is_leaf(node):
                tree.expand(node)

            if node.children:
                sim_node = max(node.children,
                               key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
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
        while not tree.is_leaf(node):
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
        return node

    def _call_verifier(self, node: SearchNode) -> None:
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score
        if hasattr(self.verifier, "true_value"):
            node.true_value = self.verifier.true_value(node)

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent


class RandomAllocationMCTS:
    """
    MCTS where each node is verified independently with probability p.
    This is an uninformed baseline that achieves sub-linear calls but no
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
            if tree.is_leaf(node):
                tree.expand(node)

            if node.children:
                sim_node = max(node.children,
                               key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
            else:
                sim_node = node

            if self.rng.random() < self.p:
                self._call_verifier(sim_node)
                verifier_calls += 1
            else:
                if hasattr(self.verifier, "true_value"):
                    sim_node.true_value = self.verifier.true_value(sim_node)
                if sim_node.parent and sim_node.parent.estimated_value is not None:
                    sim_node.proxy_value = sim_node.parent.estimated_value
                else:
                    sim_node.proxy_value = 0.5

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
        while not tree.is_leaf(node):
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.uct_score(node.visit_count, self.cfg.c_ucb))
        return node

    def _call_verifier(self, node: SearchNode) -> None:
        score = self.verifier.score(node)
        node.verifier_called = True
        node.verifier_value = score
        if hasattr(self.verifier, "true_value"):
            node.true_value = self.verifier.true_value(node)

    def _backpropagate(self, node: SearchNode, value: float) -> None:
        current: Optional[SearchNode] = node
        while current is not None:
            current.update(value)
            current = current.parent


class BestOfN:
    """
    Best-of-N baseline: generate N independent solutions and return the
    highest-scoring one according to the verifier.

    No tree structure; the 'budget' is split equally across N candidates.
    This is the simplest test-time scaling strategy (Brown et al. 2024).
    """

    def __init__(self, verifier: Verifier, rng: np.random.Generator) -> None:
        self.verifier = verifier
        self.rng = rng

    def search(self, tree: SearchTree, budget: int) -> SearchResult:
        """Sample 'budget' independent leaves and pick the best-scored one."""
        verifier_calls = 0
        best_score = -1.0
        best_node = None

        leaves = []
        for _ in range(budget):
            # Simulate an independent path from root to a random leaf
            node = tree.root
            for _ in range(tree.max_depth):
                if not node.children:
                    tree.expand(node)
                if node.children:
                    node = self.rng.choice(node.children)  # type: ignore[arg-type]
                else:
                    break
            leaves.append(node)
            score = self.verifier.score(node)
            verifier_calls += 1
            node.verifier_called = True
            node.verifier_value = score
            node.visit_count += 1
            node.total_value += score

            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            best_node = tree.root

        _, oracle_best = tree.best_true_leaf()
        best_true = best_node.true_value or 0.0
        final_regret = max(0.0, oracle_best - best_true) * budget  # cumulative proxy

        step_records = [
            StepRecord(
                step=budget,
                node_id=best_node.node_id,
                depth=best_node.depth,
                verifier_called=True,
                estimated_value=best_score,
                true_value=best_true,
                oracle_best_true=oracle_best,
                cumulative_regret=final_regret,
                total_verifier_calls=verifier_calls,
            )
        ]

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
