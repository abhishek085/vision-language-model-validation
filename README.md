# vision-language-model-validation

A small local playground for trying out **open-source vision-language models** on a
Mac (Apple Silicon) or PC. Upload an image/PDF, give a prompt (or JSON schema),
see the model's output — all running locally.

Ships with two runners:

| Model | What it does | Size | Prompt |
|-------|--------------|------|--------|
| **SmolVLM-256M** | Free-text Q&A about an image | ~0.5 GB | natural language |
| **Datalab LIFT** ([`datalab-to/lift`](https://huggingface.co/datalab-to/lift)) | Schema-constrained JSON extraction from PDFs/images | ~20 GB (9B, bf16) | JSON Schema |

Start with SmolVLM to confirm everything works — it loads in seconds.

## Setup

Requires Python ≥ 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync                  # create .venv and install deps
cp .env.example .env     # optional: pin TORCH_DEVICE etc.
./run.sh                 # or: uv run python -m app.server
```

Open <http://127.0.0.1:8000>.

## Usage

1. Pick a model from the dropdown.
2. Upload an image (or PDF, for LIFT).
3. Edit the prompt / JSON schema.
4. **Run**. The first run for a model downloads + loads its weights (slow);
   later runs reuse the loaded model.

## ⚠️ Memory note for LIFT on a 24 GB Mac

LIFT is ~9B params in bf16 (~18–20 GB). On a 24 GB machine it *can* run on the
`mps` backend but will lean on swap and be slow. If it runs out of memory, set
`TORCH_DEVICE=cpu` in `.env` (slower still, but more headroom). SmolVLM is
unaffected.

## Adding another VLM

Each model is a `VLMRunner` (see [`vlm/base.py`](vlm/base.py)). To add one:

1. Copy [`vlm/runners/smolvlm_runner.py`](vlm/runners/smolvlm_runner.py) to a new file.
2. Change `name`, `title`, the model id, and the chat formatting in `run()`.
3. Register the class in [`vlm/registry.py`](vlm/registry.py).

The UI and `/api` endpoints pick it up automatically. Set `prompt_is_json = True`
for schema-style models, and `accepts = ("image", "pdf")` if it takes PDFs.

## Layout

```
app/
  server.py        Flask app: serves the UI + /api/models, /api/run
  static/index.html  the UI (vanilla HTML/JS)
vlm/
  base.py          VLMRunner contract + RunResult
  device.py        mps/cuda/cpu selection (honors TORCH_DEVICE)
  registry.py      name -> runner, add new models here
  runners/         one file per model
samples/           example schemas / inputs
```

## Config (`.env`)

| Var | Meaning |
|-----|---------|
| `TORCH_DEVICE` | `mps` / `cpu` / `cuda` (auto-detected if unset) |
| `MAX_OUTPUT_TOKENS` | cap LIFT output length |
| `MODEL_CHECKPOINT` | override the LIFT checkpoint |
| `PORT` | web server port (default 8000) |
| `HF_HOME` | HuggingFace weight cache location |
