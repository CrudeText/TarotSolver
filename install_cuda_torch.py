#!/usr/bin/env python3
"""
Reinstall PyTorch with CUDA support (cu128). Use this if the app reports
"CPU" only and you have an NVIDIA GPU (e.g. RTX 4070).

  python install_cuda_torch.py

Then run the app again (python run.py or python -m tarot_gui.main).
Python 3.14 requires cu128; older Python can use cu121 or cu128.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    py = sys.executable
    # cu128 has wheels for Python 3.10–3.14 on Windows; cu121 has no 3.14
    index = "https://download.pytorch.org/whl/cu128"
    print("Reinstalling torch with CUDA 12.8 (index:", index, ")")
    subprocess.check_call([
        py, "-m", "pip", "uninstall", "torch", "-y",
    ])
    subprocess.check_call([
        py, "-m", "pip", "install", "torch",
        "--index-url", index,
    ])
    print("Checking CUDA...")
    r = subprocess.run(
        [py, "-c", (
            "import torch; "
            "ok = torch.cuda.is_available(); "
            "print('torch.cuda.is_available():', ok); "
            "print('GPU:', torch.cuda.get_device_name(0) if ok else 'N/A'); "
            "exit(0 if ok else 1)"
        )],
        cwd=root,
    )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
