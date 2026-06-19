"""Base contract every model runner implements.

A runner wraps exactly one vision-language model. The web app discovers
runners through the registry, lazily calls ``load()`` on first use, then
``run()`` per request.

Keep heavy imports (torch, transformers, lift) *inside* the methods so that
importing a runner module stays cheap — the registry imports every runner at
startup just to list them in the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """What a single inference call returns."""

    output: str                              # text/JSON string shown in the UI
    meta: dict[str, Any] = field(default_factory=dict)  # device, timing, model id...


class VLMRunner:
    """Subclass this to add a new model. See ``runners/smolvlm_runner.py``."""

    # --- identity + UI hints (override these) ---
    name: str = "base"                # stable id used by the API/dropdown
    title: str = "Base runner"        # human label in the dropdown
    description: str = ""             # one-liner shown under the dropdown
    accepts: tuple[str, ...] = ("image",)   # file kinds: "image" and/or "pdf"
    prompt_label: str = "Prompt"      # label for the text box
    prompt_default: str = ""          # prefilled text
    prompt_is_json: bool = False      # True -> validate + pretty-print as JSON (e.g. a schema)
    requires_file: bool = True

    def load(self) -> None:
        """Load weights into memory. Called once, lazily. Must be idempotent."""
        raise NotImplementedError

    def run(self, file_path: str | None, prompt: str, **opts: Any) -> RunResult:
        """Run inference on one (document, prompt) pair."""
        raise NotImplementedError

    def info(self) -> dict[str, Any]:
        """Serializable description for the /api/models endpoint."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "accepts": list(self.accepts),
            "prompt_label": self.prompt_label,
            "prompt_default": self.prompt_default,
            "prompt_is_json": self.prompt_is_json,
            "requires_file": self.requires_file,
        }
