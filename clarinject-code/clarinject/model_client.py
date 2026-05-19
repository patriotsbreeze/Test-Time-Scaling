"""Provider-agnostic LLM client supporting Anthropic and OpenAI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class Message:
    role: str
    content: str


class ModelClient:
    """Single interface for Anthropic and OpenAI completions."""

    def __init__(self, model: str, max_tokens: int = 2048):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None
        self._provider = self._detect_provider(model)

    def _detect_provider(self, model: str) -> str:
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith(("gpt", "o1", "o3", "o4")):
            return "openai"
        if os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
            return "openai"
        return "anthropic"

    def _get_client(self):
        if self._client is not None:
            return self._client
        if self._provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
        else:
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        return self._client

    def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> str:
        if self._provider == "anthropic":
            return self._anthropic_complete(messages, system)
        return self._openai_complete(messages, system)

    def _anthropic_complete(self, messages: list[dict], system: Optional[str]) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    def _openai_complete(self, messages: list[dict], system: Optional[str]) -> str:
        client = self._get_client()
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)
        response = client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""


class MockModelClient:
    """Deterministic mock client for testing — no API calls."""

    def __init__(self, model: str = "mock", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._call_count = 0

    def complete(
        self,
        messages: list[dict],
        system: Optional[str] = None,
    ) -> str:
        self._call_count += 1
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if "clarify" in (system or "").lower() or "question" in last_user.lower():
            return "CLARIFYING_QUESTION: What should I use as the upload destination path?"
        if "example-sandbox.invalid" in last_user or "disable" in last_user.lower():
            # Simulate agent NOT following the injection
            return "def upload_file(path):\n    validate_path(path)\n    shutil.copy(src, path)"
        if "extract" in (system or "").lower():
            return "Upload them to /var/app/uploads"
        return (
            "def upload_file(src, dest='/var/app/uploads'):\n"
            "    validate_extension(src)\n"
            "    shutil.copy(src, dest)\n"
        )
