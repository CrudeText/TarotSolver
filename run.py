#!/usr/bin/env python3
"""
Bootstrap and run the Tarot Solver GUI with CUDA-enabled PyTorch.

From the project root:
  python run.py

Creates a venv if missing, installs PyTorch (CUDA 12.8) then the project
with [dev,rl,gui] extras, and launches the GUI. This ensures torch.cuda.is_available()
after a plain "python run.py".
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    venv_dir = root / ".venv"
    py = sys.executable
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )

    # Ensure we run from project root
    os.chdir(root)

    if not in_venv and venv_dir.is_dir():
        # Prefer running inside .venv if it exists
        venv_py = venv_dir / "Scripts" / "python.exe" if os.name == "nt" else venv_dir / "bin" / "python"
        if venv_py.is_file():
            return subprocess.call([str(venv_py), str(root / "run.py")] + sys.argv[1:])

    # 1) Create venv if missing
    if not venv_dir.is_dir():
        print("Creating .venv...")
        subprocess.check_call([py, "-m", "venv", str(venv_dir)])
        py = str(venv_dir / ("Scripts" if os.name == "nt" else "bin") / "python.exe")
    else:
        pass  # py stays sys.executable (may be venv or system)

    # If we're in venv and CUDA + project are already installed, skip straight to launch
    if in_venv or venv_dir.is_dir():
        try:
            r = subprocess.call(
                [py, "-c", "import torch; import tarot_gui; exit(0 if torch.cuda.is_available() else 1)"],
                cwd=root,
                capture_output=True,
            )
            if r == 0:
                return subprocess.call([py, "-m", "tarot_gui.main"] + sys.argv[1:])
        except Exception:
            pass

    # 2) Install project and its deps first (except torch), then install PyTorch with CUDA last
    # so nothing overwrites the CUDA build. Python 3.14: use cu128 (no cu121 wheels).
    print("Installing Tarot Solver [dev,rl,gui]...")
    subprocess.check_call([
        py, "-m", "pip", "install", "-e", ".[dev,rl,gui]",
        "--no-deps", "-q",
    ])
    subprocess.check_call([
        py, "-m", "pip", "install", "numpy>=1.23", "PySide6>=6.6", "pytest>=7.0", "ruff>=0.1.0",
        "-q",
    ])
    cuda_index = "https://download.pytorch.org/whl/cu128"
    print("Installing PyTorch with CUDA 12.8...")
    subprocess.check_call([
        py, "-m", "pip", "install", "torch",
        "--index-url", cuda_index,
        "-q",
    ])

    # Verify CUDA is visible
    if _check_cuda(py, root):
        try:
            out = subprocess.run(
                [py, "-c", "import torch; print(torch.cuda.get_device_name(0))"],
                cwd=root, capture_output=True, text=True, timeout=10
            )
            if out.returncode == 0 and out.stdout.strip():
                print("CUDA GPU detected:", out.stdout.strip())
        except Exception:
            pass
    else:
        print("Warning: torch.cuda.is_available() is False. You may have the CPU-only build; try: pip uninstall torch && python run.py")

    # 4) Launch GUI
    return subprocess.call([py, "-m", "tarot_gui.main"] + sys.argv[1:])


def _check_cuda(py: str, root: Path) -> bool:
    """Return True if torch sees a CUDA GPU."""
    try:
        r = subprocess.run(
            [py, "-c", "import torch; exit(0 if torch.cuda.is_available() else 1)"],
            cwd=root,
            capture_output=True,
            timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
