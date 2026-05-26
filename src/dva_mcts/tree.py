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
    verifier_value: Optional[float] = None  # observed noisy score when called
    true_value: Optional[float] = None      # ground truth (oracle, for evaluation only)

    # NOTE: proxy_value is intentionally NOT stored persistently.
    # The dynamic Lipschitz proxy is recomputed fresh each step via
    # get_dynamic_estimate(), ensuring the error bound tau_t applies at
    # the current step t (see Section 4.3 and Lemma A.1).

    @property
    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def estimated_value(self) -> Optional[float]:
        """Best static estimate: verifier > mean > None.

        For the dynamic Lipschitz proxy (ancestor-inherited score),
        use get_dynamic_estimate(L) instead.  This property is kept
        for backward-compatibility with logging and reporting code.
        """
        if self.verifier_value is not None:
            return self.verifier_value
        if self.visit_count > 0:
            return self.mean_value
        return None

    # ── Dynamic proxy ──────────────────────────────────────────────────────

    def get_dynamic_estimate(self, L: float) -> Optional[float]:
        """Dynamic Lipschitz proxy estimate, recomputed fresh each call.

        Returns
        -------
        float | None
            - If this node has been verified: its stored verifier_value.
            - Else: nearest verified ancestor's verifier_value (neutral
              Lipschitz midpoint; V(s) lies in [V_anc ± L·Δd]).
            - None if no verified node exists on the path to the root.

        Recomputing at each step (rather than caching proxy_value) ensures
        the estimation error |V(s) - v̂_t(s)| ≤ L·Δd_t(s) reflects the
        *current* tree coverage.  As ancestors are verified over time,
        Δd_t decreases automatically—no extra verifier calls required.
        """
        if self.verifier_called and self.verifier_value is not None:
            return self.verifier_value
        ancestor = self.nearest_verified_ancestor()
        if ancestor is None:
            return None  # no information available yet
        # Neutral midpoint of [V_anc - L·Δd, V_anc + L·Δd]
        return ancestor.verifier_value

    def depth_gap_to_verified_ancestor(self) -> int:
        """Return depth gap to the nearest verified ancestor (0 if verified)."""
        if self.verifier_called:
            return 0
        ancestor = self.nearest_verified_ancestor()
        if ancestor is None:
            return self.depth  # gap to root
        return self.depth - ancestor.depth

    def uct_score(self, parent_visits: int, c_ucb: float) -> float:
        """Standard UCT score using static estimated_value."""
        if self.visit_count == 0:
            return float("inf")
        exploitation = self.estimated_value if self.estimated_value is not None else 0.0
        exploration = c_ucb * math.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploitation + exploration

    def uct_score_dynamic(
        self,
        parent_visits: int,
        c_ucb: float,
        L: float,
    ) -> float:
        """UCT score with dynamic Lipschitz proxy estimate.

        Uses get_dynamic_estimate(L) for the exploitation term, ensuring
        the score reflects the latest verified information from ancestors.
        Unverified nodes inherit their nearest verified ancestor's score,
        which is updated automatically as the tree fills in.

        Parameters
        ----------
        parent_visits : int
            Visit count of the parent node (for UCT log term).
        c_ucb : float
            UCT exploration constant (sqrt(2) by default).
        L : float
            Current Lipschitz constant estimate (oracle or adaptive L̂).
        """
        if self.visit_count == 0:
            return float("inf")
        est = self.get_dynamic_estimate(L)
        exploitation = est if est is not None else 0.0
        exploration = c_ucb * math.sqrt(math.log(max(parent_visits, 1)) / self.visit_count)
        return exploitation + exploration

    # ── Tree traversal helpers ─────────────────────────────────────────────

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
        """True iff the node has no children OR is at maximum depth."""
        return node.depth >= self.max_depth or len(node.children) == 0

    def at_max_depth(self, node: SearchNode) -> bool:
        """True iff the node is at the maximum tree depth."""
        return node.depth >= self.max_depth

    def all_leaves(self) -> List[SearchNode]:
        """Return all currently expanded leaf nodes."""
        leaves = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            if self.at_max_depth(n):
                leaves.append(n)
            elif not n.children:
                leaves.append(n)
            else:
                stack.extend(n.children)
        return leaves

    def best_true_leaf(self) -> Tuple[SearchNode, float]:
        """Oracle: return the visited node with the highest true value."""
        visited = [n for n in self._node_registry.values() if n.visit_count > 0]
        if not visited:
            return self.root, self.root.true_value or 0.0
        best = max(visited, key=lambda n: n.true_value or 0.0)
        return best, best.true_value or 0.0

    def best_estimated_leaf(self, L: float = 0.0) -> Tuple[SearchNode, float]:
        """Return the visited node with the highest dynamic estimate."""
        visited = [n for n in self._node_registry.values() if n.visit_count > 0]
        if not visited:
            return self.root, 0.0
        best = max(
            visited,
            key=lambda n: n.get_dynamic_estimate(L) if n.get_dynamic_estimate(L) is not None else -1.0,
        )
        est = best.get_dynamic_estimate(L)
        return best, est if est is not None else 0.0

    def n_verified(self) -> int:
        """Return the number of nodes that have been verified."""
        return sum(1 for n in self._node_registry.values() if n.verifier_called)

    def __len__(self) -> int:
        return len(self._node_registry)
