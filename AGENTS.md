# AGENTS.md

## Quick reference

```sh
# Activate environment (required for all commands)
conda activate open-webui-openai-compatible

# Lint
ruff check src/ tests/

# Unit tests (no credentials needed)
python -m pytest tests/test_translator.py tests/test_app.py -v

# Integration tests (requires real credentials exported BEFORE pytest starts)
OPEN_WEBUI_URL=https://your-open-webui-instance.example.com USER_TOKEN=<jwt> python -m pytest tests/integration/ -v

# Run server
cp .env.example .env  # then edit with real values
uvicorn src.main:app --port 8000
```

## Architecture

Single-package FastAPI proxy. Source lives in `src/`, no sub-packages.

| Module | Role |
|--------|------|
| `src/settings.py` | Pydantic Settings singleton — **instantiated at import time** |
| `src/client.py` | Async httpx wrapper; talks to Open WebUI `/api/models` and `/api/chat/completions` |
| `src/translator.py` | Converts Open WebUI model list → OpenAI `/v1/models` schema |
| `src/main.py` | FastAPI app with 3 routes: `/health`, `/v1/models`, `/v1/chat/completions` |
| `src/models.py` | Pydantic response types (OpenAI schema shapes) |

Request flow: client → proxy route → `WebClient` → upstream Open WebUI → translator (models only) → response.

Chat completions are pass-through (no translation). Streaming uses SSE passthrough via `StreamingResponse`.

## Critical gotcha: settings singleton

`src/settings.py` runs `settings = Settings()` at **module level**. This means:

- Importing _any_ `src` module triggers settings validation.
- If `OPEN_WEBUI_URL` or `USER_TOKEN` are missing, the import crashes with `ValidationError`.
- Tests handle this via `os.environ.setdefault()` at the top of `tests/conftest.py` — this runs before any `src` import during collection.
- The root conftest also has an `autouse` fixture that monkeypatches env vars for each test.

**When adding new test files**: always ensure `tests/conftest.py` is loaded first (pytest does this automatically for files under `tests/`). Never import from `src` at the module level of a test file without the conftest guard.

## Testing

- **Unit tests** (`tests/test_translator.py`, `tests/test_app.py`): Mock the `WebClient` via `unittest.mock.patch("src.main.WebClient")`. No network calls.
- **Integration tests** (`tests/integration/`): Use real credentials. Skip automatically when env vars are test defaults. The skip guard reads from the already-instantiated `settings` singleton, not raw env vars.
- Integration tests need real env vars **exported before pytest starts** (not just in `.env`) because `os.environ.setdefault` in the root conftest won't overwrite pre-existing vars.

## Upstream endpoint mapping

The proxy does **not** use Open WebUI's `/v1/*` paths (those require API keys, not JWTs).

| Proxy route | Upstream path | Auth |
|-------------|--------------|------|
| `GET /v1/models` | `GET /api/models` | Bearer JWT |
| `POST /v1/chat/completions` | `POST /api/chat/completions` | Bearer JWT |

## Constraints

- No hardcoded URLs anywhere in `src/` — all from `settings`.
- Error responses must use OpenAI JSON format: `{"error": {"message", "type", "code"}}`.
- Never expose `OPEN_WEBUI_URL` or `USER_TOKEN` values in error messages or logs.
- Only 3 routes exist. No `/v1/models/{id}`, no embeddings, no CORS middleware.
