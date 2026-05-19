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

    # Lipschitz constant estimate (set to true L or estimated online)
    lipschitz_L: float = 0.35

    # Sub-Gaussian noise bound (used in proxy computation)
    sigma_max: float = 0.05

    # Search tree geometry
    branching_factor: int = 2
    max_depth: int = 12

    # Whether to call verifier at depth-0 nodes always (warm-start)
    always_verify_root: bool = True

    def threshold(self, t: int) -> float:
        """Budget-sensitive threshold tau_t = gamma / t^alpha."""
        return self.gamma / max(t, 1) ** self.alpha


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
    run_best_of_n: bool = True

    # Output
    results_dir: str = "results"
    save_raw: bool = True

    # DVA-specific sweep
    alpha_values: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0])
    gamma_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
