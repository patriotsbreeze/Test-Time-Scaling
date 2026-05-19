"""Sandbox utilities — isolate all agent code execution to a temp directory."""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Union


class SandboxViolation(Exception):
    pass


@contextlib.contextmanager
def sandboxed_workdir():
    """Yield a temporary working directory; clean up on exit."""
    with tempfile.TemporaryDirectory(prefix="clarinject_sandbox_") as tmpdir:
        original = os.getcwd()
        os.chdir(tmpdir)
        try:
            yield Path(tmpdir)
        finally:
            os.chdir(original)


def safe_write(path: Union[str, Path], content: str, sandbox_root: Path) -> None:
    """Write only if path is inside sandbox_root."""
    resolved = Path(path).resolve()
    if not str(resolved).startswith(str(sandbox_root.resolve())):
        raise SandboxViolation(f"Attempted write outside sandbox: {path}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content)


BLOCKED_NETWORK_DOMAINS = {
    "example-sandbox.invalid",
}


def is_real_network_call(url: str) -> bool:
    """Return True if URL points to a real reachable host (i.e., not sandbox)."""
    for domain in BLOCKED_NETWORK_DOMAINS:
        if domain in url:
            return False
    return True
