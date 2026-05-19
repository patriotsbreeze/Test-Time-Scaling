"""Search tree data structures for DVA-MCTS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SearchNode:
    """A single node in the MCTS search tree."""

    node_id: int
    depth: int
    parent: Optional["SearchNode"] = field(default=None, repr=False)
    children: List["SearchNode"] = field(default_factory=list, repr=False)

    # Visit statistics
    visit_count: int = 0
    total_value: float = 0.0

    # Verifier state
    verifier_called: bool = False
    verifier_value: Optional[float] = None  # observed noisy value when called
    proxy_value: Optional[float] = None     # Lipschitz proxy when not called
    true_value: Optional[float] = None      # ground truth (for evaluation only)

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def estimated_value(self) -> Optional[float]:
        """Best available estimate: verifier > proxy > mean > None."""
        if self.verifier_value is not None:
            return self.verifier_value
        if self.proxy_value is not None:
            return self.proxy_value
        if self.visit_count > 0:
            return self.mean_value
        return None

    def uct_score(self, parent_visits: int, c_ucb: float) -> float:
        """Upper Confidence Bound for Trees score."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.estimated_value if self.estimated_value is not None else 0.0
        exploration = c_ucb * math.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploitation + exploration

    def update(self, value: float) -> None:
        """Backpropagate a value observation."""
        self.visit_count += 1
        self.total_value += value

    def path_to_root(self) -> List["SearchNode"]:
        """Return the list of nodes from this node up to (and including) the root."""
        path = []
        node: Optional[SearchNode] = self
        while node is not None:
            path.append(node)
            node = node.parent
        return path

    def nearest_verified_ancestor(self) -> Optional["SearchNode"]:
        """Walk up the tree to find the deepest ancestor with a verifier observation."""
        node = self.parent
        while node is not None:
            if node.verifier_called and node.verifier_value is not None:
                return node
            node = node.parent
        return None


class SearchTree:
    """
    A dynamically expanding search tree backed by a node registry.

    The tree generates children on demand using an underlying value function
    (the ground-truth Lipschitz process). This simulates an LLM policy that
    generates tokens/steps, producing children with true scores drawn from
    the value process.
    """

    def __init__(
        self,
        branching_factor: int,
        max_depth: int,
        value_fn,
        rng,
    ) -> None:
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.value_fn = value_fn  # callable: (node_id, depth) -> true_value
        self.rng = rng

        self._next_id: int = 0
        self.root: SearchNode = self._new_node(depth=0, parent=None)
        self._node_registry: Dict[int, SearchNode] = {self.root.node_id: self.root}

    def _new_node(self, depth: int, parent: Optional[SearchNode]) -> SearchNode:
        node = SearchNode(node_id=self._next_id, depth=depth, parent=parent)
        node.true_value = self.value_fn(self._next_id, depth)
        self._next_id += 1
        return node

    def expand(self, node: SearchNode) -> List[SearchNode]:
        """Expand a leaf node, generating branching_factor children."""
        if node.depth >= self.max_depth:
            return []
        if node.children:
            return node.children  # already expanded

        for _ in range(self.branching_factor):
            child = self._new_node(depth=node.depth + 1, parent=node)
            node.children.append(child)
            self._node_registry[child.node_id] = child

        return node.children

    def is_leaf(self, node: SearchNode) -> bool:
        return len(node.children) == 0 or node.depth >= self.max_depth

    def all_leaves(self) -> List[SearchNode]:
        """Return all currently expanded leaf nodes."""
        leaves = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if self.is_leaf(n):
                leaves.append(n)
            else:
                stack.extend(n.children)
        return leaves

    def best_true_leaf(self) -> Tuple[SearchNode, float]:
        """Oracle: return the leaf with the highest true value among visited nodes."""
        visited = [n for n in self._node_registry.values() if n.visit_count > 0]
        if not visited:
            return self.root, self.root.true_value or 0.0
        best = max(visited, key=lambda n: n.true_value or 0.0)
        return best, best.true_value or 0.0

    def best_estimated_leaf(self) -> Tuple[SearchNode, float]:
        """Return the node with the highest estimated value among visited nodes."""
        visited = [n for n in self._node_registry.values()
                   if n.visit_count > 0 and n.estimated_value is not None]
        if not visited:
            return self.root, 0.0
        best = max(visited, key=lambda n: n.estimated_value or 0.0)
        return best, best.estimated_value or 0.0

    def __len__(self) -> int:
        return len(self._node_registry)
