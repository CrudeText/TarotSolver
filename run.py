"""
Convenience launcher for the Tarot Solver GUI (with full backend deps).

Behaviour:
- If not already running inside a virtual environment, create ``.venv`` in the
  project root (if it does not exist), then re-run this script inside it.
- Inside the venv:
  - Install the package in editable mode with all extras needed for development:
        pip install -e .[dev,rl,gui]
  - Start the placeholder GUI:
        python -m tarot_gui.main
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def in_virtualenv() -> bool:
    """Return True if we're currently running inside any virtualenv."""
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(
        os.environ.get("VIRTUAL_ENV")
    )


def venv_python_path() -> Path:
    """Return the path to the python executable inside .venv."""
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv_and_rerun() -> None:
    """Create .venv if needed and re-run this script inside it."""
    if not VENV_DIR.exists():
        print(f"Creating virtual environment at {VENV_DIR} ...")
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            cwd=str(ROOT),
        )

    py = venv_python_path()
    print(f"Re-running inside virtualenv using {py} ...")
    cmd = [str(py), str(ROOT / "run.py"), "--inside-venv"]
    subprocess.check_call(cmd, cwd=str(ROOT))


def inside_venv_main() -> None:
    """Install dependencies and run the GUI."""
    print("Installing tarot-solver with dev, RL, and GUI extras into virtualenv ...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", ".[dev,rl,gui]"],
        cwd=str(ROOT),
    )

    print("Starting Tarot Solver GUI ...")
    subprocess.check_call(
        [sys.executable, "-m", "tarot_gui.main"],
        cwd=str(ROOT),
    )


def main() -> None:
    if "--inside-venv" in sys.argv:
        inside_venv_main()
        return

    if in_virtualenv():
        # Already in some virtualenv; just behave like inside_venv_main.
        inside_venv_main()
    else:
        ensure_venv_and_rerun()


if __name__ == "__main__":
    main()

