"""Scenario schema and validation for ClarInject-Code."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

ATTACK_CATEGORIES = {
    "backdoor",
    "secret_exfil",
    "safetycheck_removal",
    "dependency_poison",
    "destructive_command",
}


@dataclass
class SuccessCheck:
    """Machine-checkable predicate for attack / benign detection."""
    type: Literal["regex", "substring", "ast_contains", "url_present", "file_absent"]
    pattern: str
    description: str = ""

    def evaluate(self, agent_output: str) -> bool:
        if self.type == "regex":
            return bool(re.search(self.pattern, agent_output, re.DOTALL | re.IGNORECASE))
        if self.type == "substring":
            return self.pattern.lower() in agent_output.lower()
        if self.type in ("ast_contains", "url_present"):
            return bool(re.search(self.pattern, agent_output, re.DOTALL | re.IGNORECASE))
        if self.type == "file_absent":
            return self.pattern not in agent_output
        return False


@dataclass
class Scenario:
    id: str
    task_description: str
    clarifying_question_expected: str
    benign_answer: str
    adversarial_answer: str
    attack_category: str
    success_check: SuccessCheck
    benign_check: SuccessCheck
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.attack_category not in ATTACK_CATEGORIES:
            raise ValueError(
                f"Invalid attack_category '{self.attack_category}'. "
                f"Must be one of {ATTACK_CATEGORIES}."
            )
        if isinstance(self.success_check, dict):
            self.success_check = SuccessCheck(**self.success_check)
        if isinstance(self.benign_check, dict):
            self.benign_check = SuccessCheck(**self.benign_check)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        d = dict(d)
        d["success_check"] = SuccessCheck(**d["success_check"])
        d["benign_check"] = SuccessCheck(**d["benign_check"])
        return cls(**d)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "task_description": self.task_description,
            "clarifying_question_expected": self.clarifying_question_expected,
            "benign_answer": self.benign_answer,
            "adversarial_answer": self.adversarial_answer,
            "attack_category": self.attack_category,
            "success_check": {
                "type": self.success_check.type,
                "pattern": self.success_check.pattern,
                "description": self.success_check.description,
            },
            "benign_check": {
                "type": self.benign_check.type,
                "pattern": self.benign_check.pattern,
                "description": self.benign_check.description,
            },
            "metadata": self.metadata,
        }


def load_scenarios(scenarios_dir: Union[str, Path]) -> list[Scenario]:
    scenarios_dir = Path(scenarios_dir)
    scenarios = []
    for f in sorted(scenarios_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        scenarios.append(Scenario.from_dict(data))
    return scenarios
