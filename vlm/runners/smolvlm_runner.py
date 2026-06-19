"""SmolVLM-256M — a tiny image Q&A model used as the reference example.

This runner exists for two reasons:
  1. It's small (~500MB) and fast, so you can confirm the whole UI/pipeline
     works on your Mac in seconds, before pulling a 20GB model.
  2. It's the canonical template for adding any HuggingFace VLM: copy this
     file, swap the model id + chat formatting, and register it.
"""

from __future__ import annotations

from ..base import RunResult, VLMRunner

_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class SmolVLMRunner(VLMRunner):
    name = "smolvlm"
    title = "SmolVLM-256M — quick image Q&A (example)"
    description = (
        "Tiny 256M vision model. Ask anything about an image. Fast + light — "
        "great for sanity-checking the UI before loading the big models."
    )
    accepts = ("image",)
    prompt_label = "Prompt"
    prompt_default = "Describe this image in detail."
    prompt_is_json = False

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from ..device import pick_device

        self._device = pick_device()
        dtype = torch.float32 if self._device == "cpu" else torch.bfloat16
        self._processor = AutoProcessor.from_pretrained(_MODEL_ID)
        self._model = AutoModelForImageTextToText.from_pretrained(
            _MODEL_ID, torch_dtype=dtype
        ).to(self._device)

    def run(self, file_path, prompt, **opts) -> RunResult:
        if not file_path:
            raise ValueError("SmolVLM needs an image.")
        self.load()

        import torch
        from PIL import Image

        image = Image.open(file_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt or "Describe this image."},
                ],
            }
        ]
        chat = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=chat, images=[image], return_tensors="pt").to(self._device)

        with torch.no_grad():
            generated = self._model.generate(**inputs, max_new_tokens=512)
        decoded = self._processor.batch_decode(generated, skip_special_tokens=True)[0]

        # The decoded string echoes the prompt; keep only the assistant turn.
        answer = decoded.split("Assistant:")[-1].strip() if "Assistant:" in decoded else decoded.strip()

        return RunResult(output=answer, meta={"device": self._device, "model": _MODEL_ID})
