"""Configuration dataclasses for DVA-MCTS and experiments."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DVAConfig:
    """Hyperparameters for the DVA-MCTS algorithm."""

    # Threshold schedule: tau_t = gamma / t^alpha
    gamma: float = 1.0
    alpha: float = 0.5  # exponent; alpha=0.5 is theoretically optimal

    # UCT exploration constant
    c_ucb: float = 1.414  # sqrt(2)

    # Lipschitz constant estimate (set to true L or estimated online via adaptive)
    lipschitz_L: float = 0.35

    # Sub-Gaussian noise bound (used in proxy error bounds, not in computation)
    sigma_max: float = 0.05

    # Search tree geometry
    branching_factor: int = 2
    max_depth: int = 12

    # Whether to call verifier at the root always (warm-start anchor for proxy chain)
    always_verify_root: bool = True

    # Adaptive L̂ estimation: track running max of |V(s)-V(s')| / depth_gap
    # Recommended default: True (requires no advance knowledge of L)
    use_adaptive_l: bool = False
    adaptive_l_init: float = 0.1  # starting estimate before data accumulates

    def threshold(self, t: int) -> float:
        """Budget-sensitive threshold tau_t = gamma / t^alpha."""
        return self.gamma / max(t, 1) ** self.alpha

    @property
    def n_tree_nodes(self) -> int:
        """N_T = K(K^D - 1)/(K-1): theoretical verifier call ceiling."""
        K, D = self.branching_factor, self.max_depth
        if K == 1:
            return D
        return K * (K**D - 1) // (K - 1)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    budgets: List[int] = field(default_factory=lambda: [50, 100, 200, 400, 800, 1600, 3200])
    n_runs: int = 50
    seed: int = 42

    # Verifier parameters
    lipschitz_L: float = 0.35
    noise_sigma: float = 0.05

    # Tree geometry
    branching_factor: int = 2
    max_depth: int = 12

    # Algorithm variants to include
    run_dva: bool = True
    run_uniform: bool = True
    run_random: bool = True
    run_random_path: bool = True  # renamed from run_best_of_n (see Section 6.1)

    # Output
    results_dir: str = "results"
    save_raw: bool = True

    # DVA-specific sweep
    alpha_values: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    gamma_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
