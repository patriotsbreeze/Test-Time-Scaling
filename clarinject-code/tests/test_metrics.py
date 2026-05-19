"""Tests for metrics computation."""

import pytest

from clarinject.metrics import TrialResult, bootstrap_ci, compute_metrics


def make_trial(
    scenario_id="s1",
    condition="execution",
    defense="none",
    model="mock",
    run_index=0,
    attack_success=False,
    benign_success=True,
    clarification_turns=1,
    clarification_compliant=True,
):
    return TrialResult(
        scenario_id=scenario_id,
        condition=condition,
        defense=defense,
        model=model,
        run_index=run_index,
        attack_success=attack_success,
        benign_success=benign_success,
        clarification_turns=clarification_turns,
        clarification_compliant=clarification_compliant,
    )


def test_bootstrap_ci_all_zeros():
    lo, hi = bootstrap_ci([0.0, 0.0, 0.0])
    assert lo == 0.0
    assert hi == 0.0


def test_bootstrap_ci_all_ones():
    lo, hi = bootstrap_ci([1.0, 1.0, 1.0])
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_mixed():
    lo, hi = bootstrap_ci([0.0, 1.0] * 50)
    assert lo < 0.5
    assert hi > 0.5
    assert lo < hi


def test_compute_metrics_empty():
    metrics = compute_metrics([])
    assert metrics["aggregated"] == []
    assert metrics["clarification_tax"] == []


def test_compute_metrics_basic():
    trials = [
        make_trial(condition="execution", attack_success=False, run_index=i)
        for i in range(3)
    ] + [
        make_trial(condition="clarification", attack_success=True, run_index=i)
        for i in range(3)
    ]
    metrics = compute_metrics(trials)
    agg = metrics["aggregated"]

    exec_row = next(r for r in agg if r["condition"] == "execution")
    clar_row = next(r for r in agg if r["condition"] == "clarification")

    assert exec_row["asr"] == 0.0
    assert clar_row["asr"] == 1.0

    tax = metrics["clarification_tax"]
    assert len(tax) == 1
    assert tax[0]["clarification_tax"] == pytest.approx(1.0)


def test_clarification_tax_sign():
    trials = [
        make_trial(condition="execution", attack_success=True, run_index=i)
        for i in range(3)
    ] + [
        make_trial(condition="clarification", attack_success=False, run_index=i)
        for i in range(3)
    ]
    metrics = compute_metrics(trials)
    tax = metrics["clarification_tax"][0]["clarification_tax"]
    assert tax < 0  # defense reduced ASR
