"""Unit tests for the SearchTree and SearchNode data structures."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from dva_mcts.tree import SearchNode, SearchTree


def make_tree(branching=2, depth=4):
    rng = np.random.default_rng(0)
    return SearchTree(
        branching_factor=branching,
        max_depth=depth,
        value_fn=lambda node_id, d: 0.5,
        rng=rng,
    )


class TestSearchNode:
    def test_initial_state(self):
        node = SearchNode(node_id=0, depth=0)
        assert node.visit_count == 0
        assert node.mean_value == 0.0
        assert node.estimated_value is None
        assert node.verifier_called is False

    def test_update(self):
        node = SearchNode(node_id=0, depth=0)
        node.update(0.7)
        node.update(0.3)
        assert node.visit_count == 2
        assert abs(node.mean_value - 0.5) < 1e-9

    def test_uct_unvisited_returns_inf(self):
        node = SearchNode(node_id=1, depth=1)
        assert node.uct_score(10, 1.414) == float("inf")

    def test_uct_formula(self):
        import math
        node = SearchNode(node_id=1, depth=1)
        node.verifier_value = 0.6
        node.verifier_called = True
        node.update(0.6)
        score = node.uct_score(parent_visits=10, c_ucb=1.0)
        expected = 0.6 + math.sqrt(math.log(10) / 1)
        assert abs(score - expected) < 1e-9

    def test_estimated_value_priority(self):
        node = SearchNode(node_id=0, depth=0)
        # proxy only
        node.proxy_value = 0.4
        assert node.estimated_value == 0.4
        # verifier overrides proxy
        node.verifier_value = 0.9
        node.verifier_called = True
        assert node.estimated_value == 0.9

    def test_path_to_root(self):
        root = SearchNode(node_id=0, depth=0)
        child = SearchNode(node_id=1, depth=1, parent=root)
        grandchild = SearchNode(node_id=2, depth=2, parent=child)
        path = grandchild.path_to_root()
        assert len(path) == 3
        assert path[0] is grandchild
        assert path[-1] is root

    def test_nearest_verified_ancestor_none(self):
        root = SearchNode(node_id=0, depth=0)
        child = SearchNode(node_id=1, depth=1, parent=root)
        assert child.nearest_verified_ancestor() is None

    def test_nearest_verified_ancestor_found(self):
        root = SearchNode(node_id=0, depth=0)
        root.verifier_called = True
        root.verifier_value = 0.8
        child = SearchNode(node_id=1, depth=1, parent=root)
        grandchild = SearchNode(node_id=2, depth=2, parent=child)
        anc = grandchild.nearest_verified_ancestor()
        assert anc is root


class TestSearchTree:
    def test_root_is_leaf_initially(self):
        tree = make_tree()
        assert tree.is_leaf(tree.root)

    def test_expand_creates_children(self):
        tree = make_tree(branching=3)
        children = tree.expand(tree.root)
        assert len(children) == 3
        assert not tree.is_leaf(tree.root)

    def test_expand_idempotent(self):
        tree = make_tree(branching=2)
        c1 = tree.expand(tree.root)
        c2 = tree.expand(tree.root)
        assert c1 is c2  # same list returned

    def test_expand_at_max_depth_returns_empty(self):
        tree = make_tree(branching=2, depth=2)
        # Navigate to depth 2
        tree.expand(tree.root)
        tree.expand(tree.root.children[0])
        deep_node = tree.root.children[0].children[0]
        assert deep_node.depth == 2
        result = tree.expand(deep_node)
        assert result == []

    def test_node_depths_correct(self):
        tree = make_tree(branching=2, depth=3)
        tree.expand(tree.root)
        for child in tree.root.children:
            assert child.depth == 1
            tree.expand(child)
            for gc in child.children:
                assert gc.depth == 2

    def test_best_true_leaf_after_visits(self):
        tree = make_tree(branching=2, depth=2)
        tree.expand(tree.root)
        n = tree.root.children[0]
        n.true_value = 0.9
        n.visit_count = 1
        n.total_value = 0.9
        best, val = tree.best_true_leaf()
        assert val == 0.9

    def test_len(self):
        tree = make_tree(branching=2)
        assert len(tree) == 1  # just root
        tree.expand(tree.root)
        assert len(tree) == 3  # root + 2 children
