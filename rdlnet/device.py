"""Shared PyTorch device selection."""

from __future__ import annotations

import torch


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # mps = getattr(torch.backends, "mps", None)
    # if mps is not None and mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")
