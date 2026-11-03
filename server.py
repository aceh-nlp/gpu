"""Aceh NLP model server: hosts one Hugging Face model behind a small JSON API.

Run:  ACEH_MODEL=<hf repo id> python3 server.py
Env:  ACEH_MODEL     required, e.g. google/gemma-3-4b-it (see models.json for the menu)
      ACEH_REVISION  optional model revision to pin
      ACEH_PORT      listen port, default 8000
      ACEH_TOKEN     optional bearer token required on /generate

Endpoints:
  GET  /health    -> {"ok", "model", "revision", "device", "loaded"}
  POST /generate  -> {"prompt", "max_tokens"?, "temperature"?, "top_p"?, "seed"?}
                  <- {"text", "model", "revision", "ms"}
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MAX_PROMPT_CHARS = 10_000
MAX_NEW_TOKENS = 512


class Engine:
    """Wraps one loaded model. Tests substitute a stub with the same interface."""

    def __init__(self, model_id, revision=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.revision = revision or getattr(self.model.config, "_commit_hash", None) or "main"
        self.device = str(self.model.device)

    def generate(self, prompt, max_tokens, temperature, top_p, seed=None):
        import torch

        if seed is not None:
            torch.manual_seed(seed)
        if self.tokenizer.chat_template:
            inputs = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True, return_tensors="pt",
            ).to(self.model.device)
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def validate_request(body):
    """Returns (params, error). Pure function so tests cover it without a model."""
    if not isinstance(body, dict):
        return None, "Body must be a JSON object"
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "'prompt' must be a non-empty string"
    if len(prompt) > MAX_PROMPT_CHARS:
        return None, f"'prompt' exceeds {MAX_PROMPT_CHARS} characters"
    try:
        max_tokens = int(body.get("max_tokens", 200))
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 0.9))
        seed = body.get("seed")
        if seed is not None:
            seed = int(seed)
    except (TypeError, ValueError):
        return None, "max_tokens/temperature/top_p/seed must be numbers"
    if not 1 <= max_tokens <= MAX_NEW_TOKENS:
        return None, f"'max_tokens' must be 1..{MAX_NEW_TOKENS}"
    if not 0 <= temperature <= 2 or not 0 < top_p <= 1:
        return None, "'temperature' in [0,2], 'top_p' in (0,1]"
    return {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
            "top_p": top_p, "seed": seed}, None


def build_handler(engine, token=None):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} {fmt % args}")

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "model": engine.model_id,
                                 "revision": engine.revision, "device": engine.device,
                                 "loaded": True})
            else:
                self._send(404, {"error": "Unknown path"})

        def do_POST(self):
            if self.path != "/generate":
                self._send(404, {"error": "Unknown path"})
                return
            if token and self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "Missing or wrong bearer token"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, {"error": "Invalid JSON body"})
                return
            params, err = validate_request(body)
            if err:
                self._send(400, {"error": err})
                return
            start = time.time()
            try:
                with lock:
                    text = engine.generate(**params)
            except Exception as exc:  # surface, never crash the server
                self._send(500, {"error": f"Generation failed: {exc}"})
                return
            self._send(200, {"text": text, "model": engine.model_id,
                             "revision": engine.revision,
                             "ms": int((time.time() - start) * 1000)})

    return Handler


def main():
    model_id = os.environ.get("ACEH_MODEL")
    if not model_id:
        raise SystemExit("Set ACEH_MODEL to a Hugging Face model id "
                         "(see models.json), e.g.\n"
                         "  ACEH_MODEL=google/gemma-3-4b-it python3 server.py")
    port = int(os.environ.get("ACEH_PORT", "8000"))
    print(f"Loading {model_id} ...", flush=True)
    engine = Engine(model_id, os.environ.get("ACEH_REVISION"))
    print(f"Loaded on {engine.device}; listening on 0.0.0.0:{port}", flush=True)
    server = ThreadingHTTPServer(("0.0.0.0", port),
                                 build_handler(engine, os.environ.get("ACEH_TOKEN")))
    server.serve_forever()


if __name__ == "__main__":
    main()
