# aceh-nlp/gpu

**Model hosting only.** One small server (`server.py`, standard library HTTP)
that loads one Hugging Face model and serves it to the backend. No benchmark
runners, no rental tooling, no data — datasets live in
[aceh-nlp-dataset](https://github.com/aceh-nlp/dataset), the API that
calls this server lives in
[aceh-nlp-backend](https://github.com/aceh-nlp/backend).

## Run

```sh
uv sync --locked
ACEH_MODEL=google/gemma-3-4b-it ACEH_TOKEN=<shared secret> uv run python server.py
```

`ACEH_MODEL` is **required** — there is no default; pick from `models.json`
(the registry kept from the Phase 1 screening; evidence for those runs is in
the dataset repo under `results/gpu/`). On a 48 GB A6000 everything up to the
12B models fits in bf16.

| Env | Meaning |
|---|---|
| `ACEH_MODEL` | HF model id to host (required) |
| `ACEH_REVISION` | optional revision pin |
| `ACEH_PORT` | listen port, default 8000 |
| `ACEH_TOKEN` | if set, `/generate` requires `Authorization: Bearer <token>` |

## API

- `GET /health` → `{ok, model, revision, device, loaded}`
- `POST /generate` `{prompt, max_tokens≤512, temperature, top_p, seed?}` →
  `{text, model, revision, ms}` — one generation at a time (lock), chat
  template applied automatically when the model has one.

## Tests

```sh
uv run pytest -q   # stub engine; no model download, no GPU needed
```
