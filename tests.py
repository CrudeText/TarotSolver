"""
Convenience script to run the full test suite.

Usage (from project root):

    python tests.py

Behaviour:
- Ensures `pytest` is available by installing `.[dev]` if needed.
- Runs `python -m pytest` in the project root.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def ensure_test_dependencies() -> None:
    """
    Ensure pytest and RL-related deps (numpy/torch) are available.

    We install both dev and rl extras in one go: .[dev,rl]
    """
    try:
        import pytest  # noqa: F401
        import numpy  # noqa: F401
        # torch is optional for some tests and imported conditionally
        return
    except Exception:
        pass

    print("Installing test and RL dependencies (.[dev,rl]) ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev,rl]"],
        cwd=str(ROOT),
    )


def main() -> None:
    ensure_test_dependencies()
    print("Running test suite with pytest ...")
    subprocess.check_call(
        [sys.executable, "-m", "pytest"],
        cwd=str(ROOT),
    )


if __name__ == "__main__":
    main()

