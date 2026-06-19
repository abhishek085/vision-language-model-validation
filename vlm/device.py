"""Pick the inference device once, consistently, for every runner."""

from __future__ import annotations

import os

_cached: str | None = None


def pick_device() -> str:
    """Return 'mps', 'cuda', or 'cpu'.

    Honors the TORCH_DEVICE env var if set, otherwise auto-detects
    (mps on Apple Silicon). Importing torch is deferred to here.
    """
    global _cached
    if _cached is not None:
        return _cached

    env = os.environ.get("TORCH_DEVICE")
    if env:
        _cached = env
        return _cached

    try:
        import torch

        if torch.backends.mps.is_available():
            _cached = "mps"
        elif torch.cuda.is_available():
            _cached = "cuda"
        else:
            _cached = "cpu"
    except Exception:
        _cached = "cpu"
    return _cached
