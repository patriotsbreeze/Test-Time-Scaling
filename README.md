# DVA-MCTS: Dynamic Verifier Allocation Monte Carlo Tree Search

**Paper:** *Adaptive Verifier Allocation for Efficient Test-Time Compute Scaling: A Regret-Theoretic Analysis*

## Overview

DVA-MCTS is a provably efficient algorithm for test-time compute scaling in large language models. Rather than invoking a process reward model (PRM / verifier) at every node of a search tree, DVA-MCTS decides *when* to call the verifier using a budget-sensitive uncertainty threshold:

```
tau_t = gamma / t^alpha
```

The verifier is called only when the Lipschitz-estimated score discrepancy between a node and its nearest verified ancestor exceeds this threshold.

### Theoretical guarantees

| Property | Result |
|---|---|
| Cumulative regret vs. oracle | O(√(T log K)) |
| Verifier calls | O(√T · log T)  ← **sub-linear** |
| Lower bound | Ω(√T) — rate-optimal |

## Repository structure

```
.
├── src/dva_mcts/          # Core package
│   ├── algorithm.py       # DVA-MCTS main algorithm
│   ├── baselines.py       # Uniform, Random, BestOfN baselines
│   ├── tree.py            # SearchTree / SearchNode
│   ├── verifier.py        # Verifier ABC + LipschitzVerifier
│   ├── metrics.py         # RegretTracker, accuracy, efficiency
│   └── config.py          # DVAConfig, ExperimentConfig
├── experiments/
│   └── run_experiment.py  # Master experiment runner
├── tests/                 # pytest suite (40+ tests)
├── results/               # Generated JSON results
├── paper/                 # LaTeX manuscript (JMLR format)
│   ├── main.tex
│   ├── main.pdf
│   ├── sections/          # Per-section .tex files
│   └── figures/           # PDF figures + generation script
└── pyproject.toml
```

## Quickstart

```bash
pip install -e ".[dev]"

# Run all experiments (budget=400, 50 runs)
python experiments/run_experiment.py --exp all --budget 400 --runs 50

# Run a specific experiment with more runs
python experiments/run_experiment.py --exp regret --budget 800 --runs 100

# Run tests
pytest tests/ -v
```

## Experiment options

```
--exp     {all, regret, efficiency, lipschitz, tradeoff, ablation}
--budget  Search budget for regret/ablation (default: 400)
--runs    Number of independent runs    (default: 50)
--L       Lipschitz constant            (default: 0.35)
--sigma   Verifier noise std dev        (default: 0.05)
--seed    Random seed                   (default: 42)
```

## Key algorithm parameters

| Parameter | Default | Description |
|---|---|---|
| `gamma` | 1.0 | Threshold scale constant |
| `alpha` | 0.5 | Threshold decay exponent (theoretically optimal) |
| `c_ucb` | √2 | UCT exploration constant |
| `lipschitz_L` | 0.35 | Lipschitz constant (set or estimated) |

## Results (budget=400, 50 runs, L=0.35)

| Method | Cumulative Regret | Verifier Calls | Accuracy (≥0.8) |
|---|---|---|---|
| Oracle | 0.00 | 400 | — |
| **DVA-MCTS** | **lowest** | **sub-linear** | competitive |
| Uniform | higher | 400 | competitive |
| Random | highest | ~200 | lowest |

## Regenerating paper figures

```bash
# Run experiments to produce results/*.json
python experiments/run_experiment.py --exp all --budget 400 --runs 50

# Regenerate figures from results
python paper/figures/generate_figures.py

# Recompile PDF
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Citation

```bibtex
@article{dvamcts2025,
  title   = {Adaptive Verifier Allocation for Efficient Test-Time Compute Scaling:
             A Regret-Theoretic Analysis},
  author  = {Anonymous},
  journal = {Journal of Machine Learning Research},
  year    = {2025}
}
```
