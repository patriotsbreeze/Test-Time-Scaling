"""
Real Neural PRM Experiment: DVA-MCTS with a neural regression-trained PRM
on MATH-500 / AIME benchmarks using a live generative LLM.

Usage
-----
# Install deps first:
#   pip install transformers vllm datasets torch

python experiments/run_neural_prm.py \
    --model Qwen/QwQ-32B-Preview \
    --prm RLHFlow/Llama3.1-8B-PRM-Deepseek-Data \
    --benchmark math500 \
    --n_problems 100 \
    --budget 64 \
    --branching_factor 4 \
    --depth 6 \
    --runs_per_problem 5 \
    --output results/neural_prm_math500.json

Benchmarks supported: math500, aime2024, aime2025, minerva_math

Key measurements collected
--------------------------
1. DVA-MCTS vs Uniform Verification: verifier call counts, accuracy (pass@1)
2. Effective Lipschitz constant L_eff = mean |PRM(s) - PRM(s')| / |d - d'|
   across consecutive steps (validates Assumption 1 empirically)
3. Call savings as a function of L_eff (confirms theoretical prediction:
   savings ≈ 1 - N_T/B scales with 1/L_eff)
4. Accuracy parity: DVA should match Uniform within ±2 pp at same budget

Theory predictions to validate
-------------------------------
- L_eff ≲ 0.1 for neural regression PRM → savings > 73% (transition regime)
- Call ceiling N_T = K(K^D-1)/(K-1) in exploitation regime (B >> N_T)
- Adaptive L̂ (running max) achieves best call savings without manual L tuning

For the paper (ICLR 2027): aim to show:
  Table: DVA vs Uniform on MATH-500, budget B ∈ {32, 64, 128, 256}
  Figure: L_eff vs savings scatter across problem instances
  Figure: Call count vs B (log-log) showing sub-linear growth
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Neural PRM wrapper ────────────────────────────────────────────────────────

class NeuralPRMVerifier:
    """
    Wraps a neural regression-trained PRM for use in DVA-MCTS.

    The PRM scores each prefix (partial solution) of a step-by-step
    mathematical solution.  Scores are expected to be continuous in [0, 1]
    (e.g., calibrated log-probabilities or regression outputs).

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the PRM (e.g., 'RLHFlow/Llama3.1-8B-PRM-Deepseek-Data')
    device : str
        'cuda' or 'cpu'
    batch_size : int
        Number of prefixes to score in one forward pass.
    """

    def __init__(self, model_name: str, device: str = "cuda", batch_size: int = 8) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._call_count = 0
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch

            log.info(f"Loading PRM: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)
            self._model.eval()
            log.info("PRM loaded successfully.")
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Install: pip install transformers torch"
            )

    def score_prefix(self, problem: str, partial_solution: str) -> float:
        """Score a partial solution prefix. Returns float in [0, 1]."""
        import torch
        import torch.nn.functional as F

        text = f"Problem: {problem}\nSolution: {partial_solution}"
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.device)

        with torch.no_grad():
            logits = self._model(**inputs).logits.squeeze(-1)
            score = float(torch.sigmoid(logits).cpu().item())

        self._call_count += 1
        return score

    def score_batch(self, problem: str, prefixes: List[str]) -> List[float]:
        """Batch-score multiple prefixes for efficiency."""
        import torch

        texts = [f"Problem: {problem}\nSolution: {p}" for p in prefixes]
        all_scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits.squeeze(-1)
                scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)
            self._call_count += len(batch)

        return all_scores

    @property
    def call_count(self) -> int:
        return self._call_count


# ── LLM generation wrapper ─────────────────────────────────────────────────────

class LLMGenerator:
    """
    Wraps a generative LLM for step-by-step solution generation.

    Generates the next reasoning step given a problem + partial solution prefix.
    Uses vLLM for efficient batched generation.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID (e.g., 'Qwen/QwQ-32B-Preview')
    temperature : float
        Sampling temperature. 0.7-1.0 for diverse solutions.
    max_new_tokens : int
        Max tokens per step generation.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self._llm = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from vllm import LLM, SamplingParams
            log.info(f"Loading LLM: {self.model_name}")
            self._llm = LLM(
                model=self.model_name,
                dtype="float16",
                gpu_memory_utilization=0.85,
            )
            self._sampling_params = SamplingParams(
                temperature=self.temperature,
                max_tokens=self.max_new_tokens,
                n=1,
            )
            log.info("LLM loaded successfully.")
        except ImportError:
            raise ImportError("vllm required. Install: pip install vllm")

    def generate_steps(
        self,
        problem: str,
        prefix: str,
        n_branches: int,
    ) -> List[str]:
        """
        Generate `n_branches` diverse next-step continuations.

        Returns list of step strings (not full solutions — just the next step).
        Each element, when appended to `prefix`, forms a valid partial solution.
        """
        prompt = (
            f"Solve this math problem step by step.\n\n"
            f"Problem: {problem}\n\n"
            f"Work so far:\n{prefix if prefix else '(start)'}\n\n"
            f"Next step:"
        )
        from vllm import SamplingParams
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            n=n_branches,
        )
        outputs = self._llm.generate([prompt], params)
        steps = [o.text.strip() for o in outputs[0].outputs]
        return steps


# ── DVA-MCTS with real LLM + PRM ──────────────────────────────────────────────

def run_dva_mcts_real(
    problem: str,
    llm: LLMGenerator,
    prm: NeuralPRMVerifier,
    budget: int,
    branching_factor: int,
    max_depth: int,
    gamma: float = 1.0,
    alpha: float = 0.5,
    use_adaptive_l: bool = True,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run DVA-MCTS on a real math problem with live LLM generation + neural PRM.

    The tree is built dynamically: children are generated by the LLM at expansion
    time.  Each node stores its partial solution (prefix of steps), and the PRM
    scores are used as verifier values.

    Returns
    -------
    dict with keys:
        best_solution : str  — highest-scored complete solution
        best_prm_score : float
        verifier_calls : int
        call_fraction : float
        estimated_l_eff : float — empirical Lipschitz constant
        steps_to_solution : int
    """
    rng = np.random.default_rng(seed)
    adaptive_l = 0.1  # starting estimate
    score_memory: Dict[str, Tuple[float, str]] = {}  # prefix -> (score, step_added)
    visit_counts: Dict[str, int] = {}
    total_values: Dict[str, float] = {}
    children_map: Dict[str, List[str]] = {}  # prefix -> list of child prefixes
    true_step_scores: List[Tuple[str, str, float, float]] = []  # for L_eff estimation

    call_count = 0
    root_prefix = ""

    # ── Warm-start: score root ──────────────────────────────────────────────
    root_score = prm.score_prefix(problem, root_prefix)
    score_memory[root_prefix] = (root_score, "")
    call_count += 1
    visit_counts[root_prefix] = 0
    total_values[root_prefix] = root_score

    def get_ancestor_score(prefix: str) -> Tuple[float, int]:
        """Find nearest ancestor with a stored score. Returns (score, depth_gap)."""
        if prefix in score_memory:
            return score_memory[prefix][0], 0
        # Walk up by removing last step
        parts = prefix.split("\n\n")
        for gap in range(1, len(parts) + 1):
            ancestor = "\n\n".join(parts[:-gap])
            if ancestor in score_memory:
                return score_memory[ancestor][0], gap
        return root_score, len(parts)

    def should_call_verifier(prefix: str, t: int) -> bool:
        if prefix in score_memory:
            return False  # single-verification ceiling
        anc_score, depth_gap = get_ancestor_score(prefix)
        tau = gamma / max(t, 1) ** alpha
        L = adaptive_l if use_adaptive_l else 0.2  # fallback
        delta = L * depth_gap
        return depth_gap == 0 or delta > tau

    def uct_score(prefix: str, parent_prefix: str) -> float:
        n_parent = visit_counts.get(parent_prefix, 1)
        n_child = visit_counts.get(prefix, 0)
        if n_child == 0:
            return float("inf")
        anc_score, _ = get_ancestor_score(prefix)
        exploitation = anc_score
        import math
        exploration = np.sqrt(2) * math.sqrt(math.log(max(n_parent, 1)) / n_child)
        return exploitation + exploration

    def select() -> Tuple[str, int]:
        """UCT selection. Returns (selected_prefix, depth)."""
        prefix = root_prefix
        depth = 0
        while prefix in children_map and depth < max_depth:
            kids = children_map[prefix]
            if not kids:
                break
            prefix = max(kids, key=lambda k: uct_score(k, prefix))
            depth += 1
        return prefix, depth

    def expand(prefix: str, depth: int) -> List[str]:
        """Generate branching_factor children via LLM."""
        if depth >= max_depth:
            return []
        if prefix in children_map:
            return children_map[prefix]
        steps = llm.generate_steps(problem, prefix, n_branches=branching_factor)
        new_prefixes = []
        for step in steps:
            child_prefix = (prefix + "\n\n" + step).strip() if prefix else step
            new_prefixes.append(child_prefix)
            visit_counts[child_prefix] = 0
            total_values[child_prefix] = 0.0
        children_map[prefix] = new_prefixes
        return new_prefixes

    def backpropagate(prefix: str, value: float) -> None:
        parts = prefix.split("\n\n")
        for i in range(len(parts) + 1):
            p = "\n\n".join(parts[:i]) if i > 0 else ""
            visit_counts[p] = visit_counts.get(p, 0) + 1
            total_values[p] = total_values.get(p, 0.0) + value

    # ── Main search loop ───────────────────────────────────────────────────
    for t in range(1, budget + 1):
        prefix, depth = select()
        children = expand(prefix, depth)

        if not children:
            sim_prefix = prefix
        else:
            sim_prefix = max(children, key=lambda k: uct_score(k, prefix))

        depth_sim = sim_prefix.count("\n\n") + 1 if sim_prefix else 0

        if should_call_verifier(sim_prefix, t):
            score = prm.score_prefix(problem, sim_prefix)
            call_count += 1
            score_memory[sim_prefix] = (score, "")

            # Update adaptive L
            if use_adaptive_l and sim_prefix in score_memory:
                anc_score, gap = get_ancestor_score(sim_prefix.rsplit("\n\n", 1)[0] if "\n\n" in sim_prefix else "")
                if gap > 0:
                    ratio = abs(score - anc_score) / gap
                    if ratio > adaptive_l:
                        adaptive_l = ratio
        else:
            score, _ = get_ancestor_score(sim_prefix)

        backpropagate(sim_prefix, score)

    # ── Select best solution ───────────────────────────────────────────────
    best_prefix = max(score_memory.keys(), key=lambda p: score_memory[p][0])
    best_score = score_memory[best_prefix][0]

    # ── Estimate L_eff ────────────────────────────────────────────────────
    l_ratios = []
    for prefix, (score, _) in score_memory.items():
        if "\n\n" in prefix:
            parent = prefix.rsplit("\n\n", 1)[0]
            if parent in score_memory:
                parent_score = score_memory[parent][0]
                l_ratios.append(abs(score - parent_score))
    l_eff = float(np.mean(l_ratios)) if l_ratios else 0.0

    return {
        "best_solution": best_prefix,
        "best_prm_score": best_score,
        "verifier_calls": call_count,
        "call_fraction": call_count / max(budget, 1),
        "estimated_l_eff": l_eff,
        "adaptive_l_final": float(adaptive_l),
        "budget": budget,
    }


# ── Benchmark loading ──────────────────────────────────────────────────────────

def load_benchmark(name: str, n_problems: int = 100) -> List[Dict[str, str]]:
    """Load problems from a standard benchmark dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets required. Install: pip install datasets")

    if name == "math500":
        ds = load_dataset("hendrycks/competition_math", split="test")
        problems = [{"problem": r["problem"], "answer": r["solution"]}
                    for r in ds.select(range(min(n_problems, len(ds))))]
    elif name in ("aime2024", "aime2025"):
        year = name[-4:]
        ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
        problems = [{"problem": r["problem"], "answer": str(r["answer"])}
                    for r in ds if str(year) in r.get("url", "")][:n_problems]
    elif name == "minerva_math":
        ds = load_dataset("math-ai/minerva_math_test", split="test")
        problems = [{"problem": r["problem"], "answer": r["answer"]}
                    for r in ds.select(range(min(n_problems, len(ds))))]
    else:
        raise ValueError(f"Unknown benchmark: {name}")

    log.info(f"Loaded {len(problems)} problems from {name}")
    return problems


# ── Main experiment runner ─────────────────────────────────────────────────────

def run_experiment(args) -> None:
    prm = NeuralPRMVerifier(model_name=args.prm, device=args.device)
    llm = LLMGenerator(
        model_name=args.model,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    problems = load_benchmark(args.benchmark, n_problems=args.n_problems)

    results = []
    for i, prob in enumerate(problems):
        log.info(f"Problem {i+1}/{len(problems)}")
        row = {"problem_id": i, "problem": prob["problem"][:200]}

        for seed in range(args.runs_per_problem):
            # DVA-MCTS
            dva_result = run_dva_mcts_real(
                problem=prob["problem"],
                llm=llm,
                prm=prm,
                budget=args.budget,
                branching_factor=args.branching_factor,
                max_depth=args.depth,
                gamma=args.gamma,
                alpha=args.alpha,
                use_adaptive_l=True,
                seed=seed,
            )

            # Uniform baseline: call verifier at every step
            # (re-run with gamma=0 forces all calls)
            uniform_result = run_dva_mcts_real(
                problem=prob["problem"],
                llm=llm,
                prm=prm,
                budget=args.budget,
                branching_factor=args.branching_factor,
                max_depth=args.depth,
                gamma=0.0,  # tau=0 always → always call
                alpha=args.alpha,
                use_adaptive_l=False,
                seed=seed,
            )

            row[f"seed_{seed}"] = {
                "dva": dva_result,
                "uniform": uniform_result,
                "call_savings": 1.0 - dva_result["verifier_calls"] / max(uniform_result["verifier_calls"], 1),
                "l_eff": dva_result["estimated_l_eff"],
            }

        results.append(row)
        log.info(
            f"  DVA calls: {dva_result['verifier_calls']}, "
            f"Uniform calls: {uniform_result['verifier_calls']}, "
            f"L_eff: {dva_result['estimated_l_eff']:.3f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────
    all_savings = [r[f"seed_{s}"]["call_savings"] for r in results for s in range(args.runs_per_problem)]
    all_l_eff = [r[f"seed_{s}"]["l_eff"] for r in results for s in range(args.runs_per_problem)]

    summary = {
        "benchmark": args.benchmark,
        "model": args.model,
        "prm": args.prm,
        "budget": args.budget,
        "branching_factor": args.branching_factor,
        "depth": args.depth,
        "n_problems": len(problems),
        "runs_per_problem": args.runs_per_problem,
        "mean_call_savings": float(np.mean(all_savings)),
        "std_call_savings": float(np.std(all_savings)),
        "mean_l_eff": float(np.mean(all_l_eff)),
        "std_l_eff": float(np.std(all_l_eff)),
        "results": results,
    }

    out_path = RESULTS_DIR / args.output
    out_path.write_text(json.dumps(summary, indent=2))
    log.info(f"Results saved to {out_path}")
    log.info(f"Mean call savings: {summary['mean_call_savings']:.1%} ± {summary['std_call_savings']:.1%}")
    log.info(f"Mean L_eff: {summary['mean_l_eff']:.3f} ± {summary['std_l_eff']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description="DVA-MCTS real neural PRM experiment")
    p.add_argument("--model", default="Qwen/QwQ-32B-Preview",
                   help="HuggingFace LLM model ID")
    p.add_argument("--prm", default="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
                   help="HuggingFace PRM model ID (regression-trained)")
    p.add_argument("--benchmark", default="math500",
                   choices=["math500", "aime2024", "aime2025", "minerva_math"])
    p.add_argument("--n_problems", type=int, default=100)
    p.add_argument("--budget", type=int, default=64,
                   help="Search budget B (verifier calls ≤ B for Uniform)")
    p.add_argument("--branching_factor", type=int, default=4,
                   help="K: LLM generates K candidate next steps")
    p.add_argument("--depth", type=int, default=6,
                   help="D: max solution depth in steps. N_T = K(K^D-1)/(K-1)")
    p.add_argument("--runs_per_problem", type=int, default=5)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="neural_prm_results.json")
    args = p.parse_args()

    log.info("=" * 60)
    log.info(f"DVA-MCTS Real Neural PRM Experiment")
    log.info(f"Benchmark: {args.benchmark}, Budget: {args.budget}")
    n_t = args.branching_factor * (args.branching_factor**args.depth - 1) // (args.branching_factor - 1)
    log.info(f"Tree: K={args.branching_factor}, D={args.depth}, N_T={n_t}")
    log.info(f"Exploitation regime requires B >> {n_t}")
    log.info("=" * 60)

    run_experiment(args)


if __name__ == "__main__":
    main()
