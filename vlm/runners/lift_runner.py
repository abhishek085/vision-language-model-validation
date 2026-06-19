"""Datalab LIFT — schema-constrained structured extraction.

Model card: https://huggingface.co/datalab-to/lift
~9-10B Qwen3.5 vision model. You give it a JSON Schema + a PDF/image and it
returns JSON constrained to that schema. We use the package's HuggingFace
backend (vLLM doesn't run on Apple Silicon).
"""

from __future__ import annotations

import json
import os

from ..base import RunResult, VLMRunner

# A sensible starting schema so the box isn't empty. Edit it freely in the UI.
_DEFAULT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_title": {"type": "string", "description": "Title or heading of the document"},
        "date": {"type": "string", "description": "Primary date on the document"},
        "total_amount": {"type": "number", "description": "Grand total / amount due, if any"},
        "currency": {"type": "string", "description": "Currency code or symbol"},
        "line_items": {
            "type": "array",
            "description": "Itemized rows, if the document has a table",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit_price": {"type": "number"},
                    "amount": {"type": "number"},
                },
            },
        },
    },
    "required": ["document_title"],
}


class LiftRunner(VLMRunner):
    name = "lift"
    title = "Datalab LIFT — structured JSON extraction (9B)"
    description = (
        "Give it a JSON Schema + a PDF/image; returns schema-constrained JSON. "
        "First run downloads ~20GB and is heavy on a 24GB Mac — be patient."
    )
    accepts = ("image", "pdf")
    prompt_label = "JSON Schema"
    prompt_is_json = True
    prompt_default = json.dumps(_DEFAULT_SCHEMA, indent=2)

    def __init__(self) -> None:
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        from ..device import pick_device

        # lift's HF backend reads TORCH_DEVICE; set it before importing.
        os.environ.setdefault("TORCH_DEVICE", pick_device())
        from lift.model import InferenceManager

        self._model = InferenceManager(method="hf")

    def run(self, file_path, prompt, **opts) -> RunResult:
        if not file_path:
            raise ValueError("lift needs a document (PDF or image).")
        try:
            schema = json.loads(prompt) if prompt.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Schema is not valid JSON: {e}") from e

        self.load()
        from lift import extract

        max_tokens = os.environ.get("MAX_OUTPUT_TOKENS")
        kwargs = {"max_output_tokens": int(max_tokens)} if max_tokens else {}
        result = extract(file_path, schema, model=self._model, **kwargs)

        # result is a BatchOutputItem(extraction, token_count, raw, error).
        extraction = getattr(result, "extraction", result)
        try:
            output = json.dumps(extraction, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            output = str(extraction)

        return RunResult(
            output=output,
            meta={
                "device": os.environ.get("TORCH_DEVICE", "?"),
                "checkpoint": "datalab-to/lift",
                "tokens": getattr(result, "token_count", None),
                "error": getattr(result, "error", None),
            },
        )
