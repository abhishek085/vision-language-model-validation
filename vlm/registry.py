"""Maps model names to runner instances.

To add a model: implement a VLMRunner subclass under ``runners/`` and append
its class to ``_RUNNER_CLASSES`` below. Nothing else needs to change — the UI
and API pick it up automatically.
"""

from __future__ import annotations

from .base import VLMRunner
from .runners.lift_runner import LiftRunner
from .runners.smolvlm_runner import SmolVLMRunner

# Order here = order in the UI dropdown. Lightweight first.
_RUNNER_CLASSES: list[type[VLMRunner]] = [
    SmolVLMRunner,
    LiftRunner,
]

# Loaded models are cached so weights are reused across requests.
_instances: dict[str, VLMRunner] = {}


def all_infos() -> list[dict]:
    """Cheap metadata for every registered runner (no weights loaded)."""
    return [cls().info() for cls in _RUNNER_CLASSES]


def get(name: str) -> VLMRunner:
    """Return the cached runner instance for ``name`` (does not load weights)."""
    if name not in _instances:
        for cls in _RUNNER_CLASSES:
            if cls.name == name:
                _instances[name] = cls()
                break
        else:
            raise KeyError(name)
    return _instances[name]
