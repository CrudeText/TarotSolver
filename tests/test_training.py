"""Smoke tests for PPO trainer (import-level only if torch is installed)."""

import importlib
import sys


def test_training_module_imports_if_torch_available():
    """
    We keep this test very light to avoid requiring torch in all environments.

    If torch is installed, importing tarot.training should succeed.
    """
    try:
        importlib.import_module("torch")  # noqa: F401
    except Exception:
        # Skip assertion if torch is not installed.
        return

    training = importlib.import_module("tarot.training")
    assert hasattr(training, "TarotPPOTrainer")

