"""Tiny Flask app: serves the HTML UI and runs VLM inference.

Single-user, local-only tool. All inference is serialized behind one lock so
we never load two large models at once and blow past memory.
"""

from __future__ import annotations

import os
import threading
import time
import traceback
import uuid
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory

# Load .env before anything reads TORCH_DEVICE etc.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from vlm import registry  # noqa: E402  (must come after load_dotenv)

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Inference is serialized: models aren't thread-safe and RAM is finite.
_RUN_LOCK = threading.Lock()

app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/models")
def models():
    return jsonify(registry.all_infos())


@app.post("/api/run")
def run():
    model_name = request.form.get("model", "")
    prompt = request.form.get("prompt", "")
    upload = request.files.get("file")

    try:
        runner = registry.get(model_name)
    except KeyError:
        return jsonify({"error": f"Unknown model '{model_name}'"}), 400

    file_path = None
    if upload and upload.filename:
        # Prefix with a uuid to avoid collisions; keep the extension.
        safe = f"{uuid.uuid4().hex}_{Path(upload.filename).name}"
        file_path = str(UPLOAD_DIR / safe)
        upload.save(file_path)

    t0 = time.time()
    try:
        with _RUN_LOCK:
            result = runner.run(file_path, prompt)
    except Exception as exc:  # surface a readable error to the UI
        traceback.print_exc()
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500

    meta = dict(result.meta)
    meta["seconds"] = round(time.time() - t0, 2)
    return jsonify({"output": result.output, "meta": meta})


def main():
    port = int(os.environ.get("PORT", "8000"))
    print(f"\n  VLM validation UI  ->  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
