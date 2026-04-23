# AGENTS.md

## Quick reference

```sh
# Activate environment (required for all commands)
conda activate open-webui-openai-compatible

# Lint
ruff check src/ tests/

# Type check
pyright src/

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
| `src/client.py` | Async httpx wrapper; accepts `base_url` and `token` via constructor (dependency injection) |
| `src/translator.py` | Model list translation, request body sanitization, Claude thinking variant logic |
| `src/main.py` | FastAPI app with 3 routes: `/health`, `/v1/models`, `/v1/chat/completions` |
| `src/models.py` | Pydantic response types (OpenAI schema shapes) — used by `translator.py` for validated serialization |

### Request flow

1. Client sends OpenAI-compatible request to proxy
2. `sanitize_chat_body()` strips non-OpenAI fields (e.g. LiteLLM's `extra_body`, `api_base`)
3. `resolve_thinking_model()` checks for `:extended`/`:adaptive` suffix, strips it, returns thinking config
4. `apply_thinking_params()` injects `thinking` param and ensures `max_tokens` is sufficient
5. `WebClient` forwards to upstream Open WebUI `/api/chat/completions`
6. Response is passed through (chat) or translated (models only)

### Streaming error handling

Streaming uses raw byte passthrough via `httpx.Response.aiter_raw()` to preserve upstream SSE framing exactly, avoiding line-by-line re-parsing overhead. The proxy uses a first-chunk pre-read pattern: it advances the upstream async generator by one iteration *before* returning `StreamingResponse`. If the upstream rejects immediately (e.g. 400), the proxy returns a proper HTTP error JSON response.

Error handling distinguishes specific failure modes:
- `httpx.HTTPStatusError` → upstream HTTP error (preserves status code)
- `httpx.ReadTimeout` / `httpx.ConnectTimeout` → 504 with `timeout_error` type
- `httpx.RemoteProtocolError` / `httpx.ReadError` → 502 connection failure
- Other exceptions → 502 generic, but **logged with exception type** for debugging

Errors that occur mid-stream (after headers are sent) emit SSE error events: `data: {"error": ...}` followed by `data: [DONE]`.

The `StreamingResponse` includes `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers to prevent reverse proxy (nginx/Caddy) buffering of SSE streams.

### Claude thinking variants

Models with `"claude"` in the ID get virtual thinking variants appended to `/v1/models`:
- `:extended` — `thinking.type=enabled` with `budget_tokens` (32k standard, 16k for Haiku)
- `:adaptive` — `thinking.type=adaptive` (not generated for Haiku, which doesn't support it)

The variant suffix is stripped before forwarding to upstream. The `thinking` param is injected *after* `sanitize_chat_body()` runs, so it is not in `OPENAI_CHAT_PARAMS` and doesn't need to be.

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
- **OpenAI SDK tests** (`tests/integration/test_openai_sdk.py`): Wire the OpenAI client through `TestClient` via `http_client=client` with `base_url="http://testserver/v1"`. Covers models, streaming, tool calls, and thinking variants.
- Integration tests need real env vars **exported before pytest starts** (not just in `.env`) because `os.environ.setdefault` in the root conftest won't overwrite pre-existing vars.

## Request body sanitization

`OPENAI_CHAT_PARAMS` in `translator.py` is a whitelist of standard OpenAI chat completion parameters. All other fields are stripped before forwarding. This prevents LiteLLM-injected fields like `extra_body`, `api_base`, and `custom_llm_provider` from reaching upstream (which would cause 400 errors from strict model providers like Bedrock).

To allow a new parameter through, add it to `OPENAI_CHAT_PARAMS`. The `thinking` param bypasses this because it's injected *after* sanitization by the thinking variant logic.

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
