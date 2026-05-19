"""
DVA-MCTS: Dynamic Verifier Allocation Monte Carlo Tree Search

A regret-optimal algorithm for efficient test-time compute scaling in LLMs.
Achieves O(sqrt(T log K)) regret against oracle allocation while invoking
the verifier only O(sqrt(T) log T) times.
"""

from .tree import SearchNode, SearchTree
from .verifier import Verifier, LipschitzVerifier, OracleVerifier
from .algorithm import DVAMCTS, SearchResult
from .baselines import UniformMCTS, RandomAllocationMCTS, BestOfN
from .metrics import RegretTracker, compute_accuracy, compute_verifier_efficiency
from .config import DVAConfig, ExperimentConfig

__version__ = "1.0.0"
__all__ = [
    "SearchNode",
    "SearchTree",
    "Verifier",
    "LipschitzVerifier",
    "OracleVerifier",
    "DVAMCTS",
    "SearchResult",
    "UniformMCTS",
    "RandomAllocationMCTS",
    "BestOfN",
    "RegretTracker",
    "compute_accuracy",
    "compute_verifier_efficiency",
    "DVAConfig",
    "ExperimentConfig",
]
