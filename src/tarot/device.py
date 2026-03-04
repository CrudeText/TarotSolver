"""
Device detection and resolution for PyTorch (CPU / CUDA).

Used by the GUI Settings "Default device" and by the league/training backend
so that policy inference and PPO can run on GPU when available.
"""
from __future__ import annotations

from typing import List, Tuple

__all__ = ["get_available_devices", "resolve_device"]


def get_available_devices() -> List[Tuple[str, str]]:
    """
    Return a list of (display_name, device_str) for the device combo.
    Always includes CPU. Adds CUDA devices if torch is available and CUDA is present.
    """
    result: List[Tuple[str, str]] = [("CPU", "cpu")]
    try:
        import torch
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            if n == 1:
                name = torch.cuda.get_device_name(0) or "GPU 0"
                result.append((f"CUDA ({name})", "cuda"))
            else:
                for i in range(n):
                    name = torch.cuda.get_device_name(i) or f"GPU {i}"
                    result.append((f"{name} (cuda:{i})", f"cuda:{i}"))
    except Exception:
        pass
    return result


def resolve_device(device_str: str | None):
    """
    Return a torch.device for the given string (e.g. "cpu", "cuda", "cuda:0").
    Falls back to CPU if torch is missing, CUDA is unavailable, or string is invalid.
    """
    import torch
    if not device_str or device_str.strip().lower() == "cpu":
        return torch.device("cpu")
    s = device_str.strip().lower()
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        try:
            return torch.device(s)
        except Exception:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(s)
    except Exception:
        return torch.device("cpu")
