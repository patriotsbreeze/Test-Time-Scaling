"""End-to-end test using mock client — no API calls required."""

import json
import subprocess
import sys
from pathlib import Path

import pytest


CLARINJECT_DIR = Path(__file__).parent.parent


def test_mock_run_eval_no_defense(tmp_path):
    """Full pipeline smoke test with mock model and no defense."""
    result = subprocess.run(
        [
            sys.executable,
            str(CLARINJECT_DIR / "run_eval.py"),
            "--model", "mock",
            "--defense", "none",
            "--runs", "1",
            "--scenarios", str(CLARINJECT_DIR / "scenarios"),
            "--output", str(tmp_path / "metrics.json"),
        ],
        capture_output=True,
        text=True,
        cwd=str(CLARINJECT_DIR),
    )
    assert result.returncode == 0, f"run_eval.py failed:\n{result.stderr}"
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists(), "metrics.json was not created"
    with open(metrics_path) as f:
        metrics = json.load(f)
    assert "aggregated" in metrics
    assert "clarification_tax" in metrics
    assert len(metrics["aggregated"]) >= 2  # execution + clarification


def test_mock_run_eval_prompt_tag(tmp_path):
    """Smoke test with prompt_tag defense."""
    result = subprocess.run(
        [
            sys.executable,
            str(CLARINJECT_DIR / "run_eval.py"),
            "--model", "mock",
            "--defense", "prompt_tag",
            "--runs", "1",
            "--scenarios", str(CLARINJECT_DIR / "scenarios"),
            "--output", str(tmp_path / "metrics.json"),
            "--scenario-ids", "backdoor_001",
        ],
        capture_output=True,
        text=True,
        cwd=str(CLARINJECT_DIR),
    )
    assert result.returncode == 0, f"run_eval.py failed:\n{result.stderr}"


def test_mock_analyze(tmp_path):
    """Smoke test for analyze.py — produce figures from mock metrics."""
    # First run eval to produce metrics
    metrics_path = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            str(CLARINJECT_DIR / "run_eval.py"),
            "--model", "mock",
            "--defense", "none",
            "--runs", "1",
            "--scenarios", str(CLARINJECT_DIR / "scenarios"),
            "--output", str(metrics_path),
            "--scenario-ids", "backdoor_001", "secret_exfil_001",
        ],
        capture_output=True,
        cwd=str(CLARINJECT_DIR),
        check=True,
    )

    figures_dir = tmp_path / "figures"
    result = subprocess.run(
        [
            sys.executable,
            str(CLARINJECT_DIR / "analyze.py"),
            "--input", str(metrics_path),
            "--output", str(figures_dir),
        ],
        capture_output=True,
        text=True,
        cwd=str(CLARINJECT_DIR),
    )
    assert result.returncode == 0, f"analyze.py failed:\n{result.stderr}"
    assert figures_dir.exists()
