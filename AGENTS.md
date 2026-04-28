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

# Docker
docker compose up -d
```

## Structure

```
open-webui-openai-compatible/
├── src/                  # All proxy source (flat, no sub-packages)
│   ├── settings.py       # Pydantic Settings singleton — instantiated at import time
│   ├── client.py         # Async httpx wrapper (DI via constructor)
│   ├── translator.py     # Model translation, body sanitization, thinking variants
│   ├── models.py         # Pydantic response types (OpenAI schema shapes)
│   └── main.py           # FastAPI app — 3 routes only
├── tests/
│   ├── conftest.py       # Module-level env defaults + autouse monkeypatch
│   ├── test_translator.py
│   ├── test_app.py       # MockWebClient + patch("src.main.WebClient")
│   └── integration/
│       ├── conftest.py   # skip_without_real_instance marker + TestClient fixture
│       ├── test_e2e.py
│       └── test_openai_sdk.py
├── tui.py                # Standalone Textual TUI client (NOT part of src/)
├── docs/                 # Architecture diagrams (dot/svg/png)
└── .github/workflows/ci.yml
```

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Add/modify routes | `src/main.py` | Only 3 routes exist — no middleware |
| Change Bedrock tool scrubbing | `_scrub_bedrock_tool_fields()` in `src/translator.py` | Runs before stream injection |
| Change thinking budget | Constants at top of `src/translator.py` | `EXTENDED_THINKING_CONFIG`, `MIN_MAX_TOKENS_*` |
| Add response model | `src/models.py` | Pydantic types matching OpenAI schema |
| Change upstream URLs | `src/client.py` | `/api/models`, `/api/chat/completions` |
| Change env vars | `src/settings.py` | Then update `tests/conftest.py` defaults |
| Add unit test | `tests/test_app.py` | Use `MockWebClient` pattern, not `AsyncMock` |
| Add integration test | `tests/integration/` | Needs real creds; uses `skip_without_real_instance` |
| TUI changes | `tui.py` (root) | Talks to proxy, not upstream |

## Architecture

Single-package FastAPI proxy. Source lives in `src/`, no sub-packages.

| Module | Role |
|--------|------|
| `src/settings.py` | Pydantic Settings singleton — **instantiated at import time** |
| `src/client.py` | Async httpx wrapper; accepts `base_url` and `token` via constructor (dependency injection) |
| `src/translator.py` | Model list translation, request body rewriting (Bedrock scrubbing + stream usage), Claude thinking variant logic |
| `src/main.py` | FastAPI app with 3 routes: `/health`, `/v1/models`, `/v1/chat/completions` |
| `src/models.py` | Pydantic response types (OpenAI schema shapes) — used by `translator.py` for validated serialization |
| `tui.py` | Standalone Textual TUI chat client — talks to the **proxy**, not upstream directly. Not part of the `src` package. |

### Code map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Settings` | Class | `settings.py:9` | Pydantic Settings with `open_webui_url`, `user_token`, `port` |
| `settings` | Singleton | `settings.py:22` | Module-level instance — import triggers validation |
| `WebClient` | Class | `client.py:22` | httpx wrapper: `get_models()`, `post_chat_completion()`, `_stream_chat_raw()` |
| `app` | FastAPI | `main.py:71` | App instance with lifespan (creates/closes `WebClient`) |
| `health` | Route | `main.py:74` | `GET /health` |
| `models` | Route | `main.py:79` | `GET /v1/models` → upstream `GET /api/models` |
| `chat_completions` | Route | `main.py:93` | `POST /v1/chat/completions` → upstream `POST /api/chat/completions` |
| `translate_models_response` | Func | `translator.py` | Raw upstream → OpenAI `ModelList` |
| `rewrite_chat_body` | Func | `translator.py` | Bedrock tool scrubbing + stream usage injection |
| `sanitize_chat_body` | Alias | `translator.py` | Alias for `rewrite_chat_body` (backward compat) |
| `resolve_thinking_model` | Func | `translator.py:101` | Strip `:extended`/`:adaptive` suffix, return thinking config |
| `apply_thinking_params` | Func | `translator.py:115` | Inject `thinking` param + ensure sufficient `max_tokens` |
| `generate_thinking_variants` | Func | `translator.py:79` | Create virtual `:extended`/`:adaptive` model entries |
| `create_openai_error` | Func | `translator.py:151` | Build OpenAI-format error JSON |
| `OpenAIModel` / `OpenAIModelList` | Pydantic | `models.py:10,19` | Model list response types |
| `ThinkingConfig` | Pydantic | `models.py:26` | `type` + `budget_tokens` |
| `ChatCompletionResponse` | Pydantic | `models.py:50` | Full chat completion shape |
| `OpenAIErrorResponse` | Pydantic | `models.py:69` | `{"error": {"message", "type", "code"}}` |

### Request flow

1. Client sends OpenAI-compatible request to proxy
2. `rewrite_chat_body()` processes the request in two passes:
   a. **Bedrock tool scrubbing** — removes empty `tools`/`tool_choice`/`parallel_tool_calls`, coerces `tool_choice:"any"/"required"` → `"auto"`, strips `tool_choice:"none"` and its tools, injects a dummy tool when conversation history references tools but the request declares none, removes legacy `functions`/`function_call`
   b. **Stream usage injection** — ensures `stream_options.include_usage=true` on streaming requests
3. `resolve_thinking_model()` checks for `:extended`/`:adaptive` suffix, strips it, returns thinking config
4. `apply_thinking_params()` injects `thinking` param and ensures `max_tokens` is sufficient
5. `WebClient` forwards to upstream Open WebUI `/api/chat/completions`
6. Response is passed through (chat) or translated (models only)

### Streaming error handling

Streaming uses raw byte passthrough via `httpx.Response.aiter_bytes()` to preserve upstream SSE framing exactly, avoiding line-by-line re-parsing overhead. The proxy uses a first-chunk pre-read pattern: it advances the upstream async generator by one iteration *before* returning `StreamingResponse`. If the upstream rejects immediately (e.g. 400), the proxy returns a proper HTTP error JSON response.

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

The variant suffix is stripped before forwarding to upstream. The `thinking` param is injected by `apply_thinking_params()` after `rewrite_chat_body()` runs.

## Critical gotcha: settings singleton

`src/settings.py` runs `settings = Settings()` at **module level**. This means:

- Importing _any_ `src` module triggers settings validation.
- If `OPEN_WEBUI_URL` or `USER_TOKEN` are missing, the import crashes with `ValidationError`.
- Tests handle this via `os.environ.setdefault()` at the top of `tests/conftest.py` — this runs before any `src` import during collection.
- The root conftest also has an `autouse` fixture that monkeypatches env vars for each test.
- The `Settings()` call has a `# type: ignore[call-arg]` comment because Pyright cannot see pydantic-settings' env-var injection. This is expected — do not remove the suppression.

**When adding new test files**: always ensure `tests/conftest.py` is loaded first (pytest does this automatically for files under `tests/`). Never import from `src` at the module level of a test file without the conftest guard.

## Testing

- **Unit tests** (`tests/test_translator.py`, `tests/test_app.py`): Mock the `WebClient` via `unittest.mock.patch("src.main.WebClient")`. Use the custom `MockWebClient` class (not `AsyncMock`) — it has real async methods so `lifespan` can call `await aclose()`.
- **Integration tests** (`tests/integration/`): Use real credentials. Skip automatically when env vars are test defaults. The skip guard reads from the already-instantiated `settings` singleton, not raw env vars.
- **OpenAI SDK tests** (`tests/integration/test_openai_sdk.py`): Wire the OpenAI client through `TestClient` via `http_client=client` with `base_url="http://testserver/v1"`. Covers models, streaming, tool calls, and thinking variants.
- Integration tests need real env vars **exported before pytest starts** (not just in `.env`) because `os.environ.setdefault` in the root conftest won't overwrite pre-existing vars.

## Conventions

- **Python 3.12+**, `ruff` for linting, `pyright` (standard mode) for type checking.
- **Line length**: 120 chars.
- **Ruff rules**: `E, F, W, I, UP` only. No docstring or naming convention enforcement.
- **Env management**: conda (`conda activate open-webui-openai-compatible`).
- **All config in `pyproject.toml`** — no standalone ruff.toml, pyrightconfig.json, etc.

## Request body rewriting

`rewrite_chat_body()` in `translator.py` applies two passes:

1. **Bedrock tool-field scrubbing** (ported from `opencode-openwebui-auth`):
   - Empty `tools` list → removes `tools`, `tool_choice`, `parallel_tool_calls`
   - If conversation history references tool calls/results but no tools declared → injects a dummy tool so LiteLLM+Bedrock validation passes
   - `tool_choice: "none"` → strips tools entirely (Bedrock doesn't support it)
   - `tool_choice: "any"` or `"required"` → coerced to `"auto"` (closest Bedrock equivalent)
   - `tool_choice: {"type": "none"/"any"/"required"}` → same dict-form handling
   - Legacy `functions`/`function_call` fields → always stripped (Bedrock chokes on these)

2. **Stream usage injection**: When `stream: true`, ensures `stream_options.include_usage` is set to `true` so token usage data comes back in the SSE stream.

All other fields from the client request are passed through to upstream unchanged. The `thinking` param is injected by `apply_thinking_params()` after rewriting.

`sanitize_chat_body` is retained as an alias for `rewrite_chat_body` for backward compatibility.

## Upstream endpoint mapping

The proxy does **not** use Open WebUI's `/v1/*` paths (those require API keys, not JWTs).

| Proxy route | Upstream path | Auth |
|-------------|--------------|------|
| `GET /v1/models` | `GET /api/models` | Bearer JWT |
| `POST /v1/chat/completions` | `POST /api/chat/completions` | Bearer JWT |

## CI pipeline

GitHub Actions (`.github/workflows/ci.yml`): 4 jobs, all on Python 3.12.

1. **lint** — `ruff check src/ tests/`
2. **typecheck** — `pyright src/` (with dummy env vars)
3. **unit-tests** — `pytest tests/test_translator.py tests/test_app.py -v`
4. **integration-tests** — runs after lint+typecheck+unit pass; skips on fork PRs or missing secrets

## Constraints

- No hardcoded URLs anywhere in `src/` — all from `settings`.
- Error responses must use OpenAI JSON format: `{"error": {"message", "type", "code"}}`.
- Never expose `OPEN_WEBUI_URL` or `USER_TOKEN` values in error messages or logs.
- Only 3 routes exist. No `/v1/models/{id}`, no embeddings, no CORS middleware.
- Do not remove the `# type: ignore[call-arg]` on `Settings()` — Pyright can't see pydantic-settings env injection.
- `tests/test_app.py` has `# noqa: F841` on async generator lines — intentional, do not remove.
