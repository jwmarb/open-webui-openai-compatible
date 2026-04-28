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
│   ├── client.py         # Async httpx wrapper — used only for /v1/models
│   ├── translator.py     # Model translation, body sanitization, thinking variants
│   ├── models.py         # Pydantic response types (OpenAI schema shapes)
│   └── main.py           # FastAPI app — 3 routes only
├── tests/
│   ├── conftest.py       # Module-level env defaults + autouse monkeypatch
│   ├── test_translator.py
│   ├── test_app.py       # MockAsyncOpenAI + MockWebClient, patches both
│   └── integration/
│       ├── conftest.py   # skip_without_real_instance marker + TestClient fixture
│       ├── test_e2e.py
│       ├── test_openai_sdk.py
│       ├── test_parallel_tool_calls.py
│       └── test_orchestrator_tool_calls.py
├── tui.py                # Standalone Textual TUI client (NOT part of src/)
├── docs/                 # Architecture diagrams (dot/svg/png)
└── .github/workflows/ci.yml
```

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Add/modify routes | `src/main.py` | Only 3 routes exist — no middleware |
| Change Bedrock tool scrubbing | `_scrub_bedrock_tool_fields()` in `src/translator.py:141` | Runs before stream injection |
| Change thinking budget | Constants at top of `src/translator.py` | `EXTENDED_THINKING_CONFIG`, `MIN_MAX_TOKENS_*` |
| Add response model | `src/models.py` | Pydantic types matching OpenAI schema |
| Change upstream URLs | `src/client.py` | `/api/models` only; chat uses openai SDK |
| Change env vars | `src/settings.py` | Then update `tests/conftest.py` defaults AND CI typecheck env |
| Add SDK-known param | `_SDK_KNOWN_PARAMS` in `src/main.py:47` | Unlisted fields silently go to `extra_body` |
| Change empty-stream retry | `_handle_streaming()` in `src/main.py:272` | Controlled by `settings.stream_empty_retry_max` |
| Add unit test | `tests/test_app.py` | Use `MockAsyncOpenAI` pattern for chat, `MockWebClient` for models |
| Add integration test | `tests/integration/` | Needs real creds; uses `skip_without_real_instance` |
| TUI changes | `tui.py` (root) | Talks to proxy, not upstream; not linted/typechecked in CI |

## Architecture

Single-package FastAPI proxy. Source lives in `src/`, no sub-packages.

| Module | Role |
|--------|------|
| `src/settings.py` | Pydantic Settings singleton — **instantiated at import time** |
| `src/client.py` | Async httpx wrapper; accepts `base_url` and `token` via constructor — **used only for `/v1/models`** |
| `src/translator.py` | Model list translation, request body rewriting (Bedrock scrubbing + stream usage), Claude thinking variant logic |
| `src/main.py` | FastAPI app with 3 routes: `/health`, `/v1/models`, `/v1/chat/completions` |
| `src/models.py` | Pydantic response types (OpenAI schema shapes) — used by `translator.py` for validated serialization |
| `tui.py` | Standalone Textual TUI chat client — talks to the **proxy**, not upstream directly. Not part of the `src` package. |

### Code map

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `Settings` | Class | `settings.py:9` | Pydantic Settings with `open_webui_url`, `user_token`, `port`, `request_timeout`, `stream_empty_retry_max`, `log_level` |
| `settings` | Singleton | `settings.py:36` | Module-level instance — import triggers validation |
| `WebClient` | Class | `client.py:17` | httpx wrapper: `get_models()`, `aclose()` |
| `app` | FastAPI | `main.py:204` | App instance with lifespan (creates/closes `WebClient` and `AsyncOpenAI`) |
| `_SDK_KNOWN_PARAMS` | Const | `main.py:47` | Whitelist of fields routed to SDK kwargs; everything else → `extra_body` |
| `_split_body_for_sdk` | Func | `main.py:59` | Separates SDK kwargs from `extra_body` based on `_SDK_KNOWN_PARAMS` |
| `_classify_upstream_error` | Func | `main.py:75` | Maps exceptions → `(message, error_type, http_status)` |
| `_stream_with_first` | Func | `main.py:141` | Yields SSE bytes for pre-read first chunk + rest; synthesizes `finish_reason` if missing |
| `_handle_streaming` | Func | `main.py:272` | Streaming path: first-chunk pre-read, empty-stream retry, SSE serialization |
| `_handle_non_streaming` | Func | `main.py:335` | Non-streaming path: direct SDK call, JSON response |
| `health` | Route | `main.py:207` | `GET /health` |
| `models` | Route | `main.py:212` | `GET /v1/models` → upstream `GET /api/models` |
| `chat_completions` | Route | `main.py:238` | `POST /v1/chat/completions` → upstream `POST /api/chat/completions` |
| `translate_models_response` | Func | `translator.py:109` | Raw upstream → OpenAI `ModelList` |
| `rewrite_chat_body` | Func | `translator.py:183` | Bedrock tool scrubbing + stream usage injection |
| `sanitize_chat_body` | Alias | `translator.py:190` | Alias for `rewrite_chat_body` (backward compat) |
| `resolve_thinking_model` | Func | `translator.py:79` | Strip `:extended`/`:adaptive` suffix, return thinking config |
| `apply_thinking_params` | Func | `translator.py:93` | Inject `thinking` param + ensure sufficient `max_tokens` |
| `generate_thinking_variants` | Func | `translator.py:57` | Create virtual `:extended`/`:adaptive` model entries |
| `create_openai_error` | Func | `translator.py:193` | Build OpenAI-format error JSON |
| `OpenAIModel` / `OpenAIModelList` | Pydantic | `models.py:10,19` | Model list response types |
| `ThinkingConfig` | Pydantic | `models.py:26` | `type` + `budget_tokens` |
| `OpenAIErrorDetail` / `OpenAIErrorResponse` | Pydantic | `models.py:33,41` | `{"error": {"message", "type", "code"}}` |

### Request flow

1. Client sends OpenAI-compatible request to proxy
2. `rewrite_chat_body()` processes the request in two passes:
   a. **Bedrock tool scrubbing** — removes empty `tools`/`tool_choice`/`parallel_tool_calls`, coerces `tool_choice:"any"/"required"` → `"auto"`, strips `tool_choice:"none"` and its tools, injects a dummy tool when conversation history references tools but the request declares none, removes legacy `functions`/`function_call`
   b. **Stream usage injection** — ensures `stream_options.include_usage=true` on streaming requests
3. `resolve_thinking_model()` checks for `:extended`/`:adaptive` suffix, strips it, returns thinking config
4. `apply_thinking_params()` injects `thinking` param and ensures `max_tokens` is sufficient
5. `_split_body_for_sdk()` separates fields into SDK kwargs (standard OpenAI params) and `extra_body` (non-standard fields like `thinking`, provider-specific pass-through)
6. `openai.AsyncOpenAI.chat.completions.create()` forwards to upstream Open WebUI `/api/chat/completions`
7. Response is passed through (chat) or translated (models only)

### Streaming error handling

Streaming uses the openai SDK's `AsyncStream[ChatCompletionChunk]` which returns parsed chunk objects. The proxy re-serializes each chunk to SSE format via `chunk.model_dump_json(exclude_unset=True)` and appends `data: [DONE]` when the stream ends.

The proxy uses a first-chunk pre-read pattern: it calls `__anext__()` on the stream *before* returning `StreamingResponse`. If the upstream rejects immediately (e.g. 400), the proxy returns a proper HTTP error JSON response.

**Empty-stream retry**: If the first `__anext__()` raises any exception (including `StopAsyncIteration`), the proxy retries up to `settings.stream_empty_retry_max` times (default 3) with exponential backoff (`1 << attempt` seconds, capped at 120s). **4xx client errors are never retried** — they short-circuit immediately. On retry exhaustion with `StopAsyncIteration`, a synthetic `finish_reason="stop"` chunk is returned as a valid 200 SSE stream.

Error handling distinguishes specific failure modes:
- `openai.APIStatusError` → upstream HTTP error (preserves status code)
- `openai.APITimeoutError` → 504 with `timeout_error` type
- `openai.APIConnectionError` → 502 connection failure
- Other exceptions → 502 generic, but **logged with exception type** for debugging

Errors that occur mid-stream (after headers are sent) emit SSE error events: `data: {"error": ...}` followed by `data: [DONE]`.

The `StreamingResponse` includes `Cache-Control: no-cache` and `X-Accel-Buffering: no` headers to prevent reverse proxy (nginx/Caddy) buffering of SSE streams.

### Streaming finish-reason guard

`_stream_with_first()` tracks whether any chunk contained a non-null `finish_reason`. If the stream ends without one, it synthesizes a `finish_reason: "stop"` chunk. This prevents clients from hanging when upstream omits the termination signal.

### Claude thinking variants

Models with `"claude"` in the ID get virtual thinking variants appended to `/v1/models`:
- `:extended` — `thinking.type=enabled` with `budget_tokens` (32k standard, 16k for Haiku)
- `:adaptive` — `thinking.type=adaptive` (not generated for Haiku, which doesn't support it)

The variant suffix is stripped before forwarding to upstream. The `thinking` param is injected by `apply_thinking_params()` after `rewrite_chat_body()` runs.

## Critical gotcha: settings singleton

`src/settings.py` runs `settings = Settings()` at **module level** (line 36). This means:

- Importing _any_ `src` module triggers settings validation.
- If `OPEN_WEBUI_URL` or `USER_TOKEN` are missing, the import crashes with `ValidationError`.
- Tests handle this via `os.environ.setdefault()` at the top of `tests/conftest.py` — this runs before any `src` import during collection.
- The root conftest also has an `autouse` fixture that monkeypatches env vars for each test.
- The `Settings()` call has a `# type: ignore[call-arg]` comment because Pyright cannot see pydantic-settings' env-var injection. This is expected — do not remove the suppression.
- The CI typecheck job injects dummy env vars (`OPEN_WEBUI_URL`, `USER_TOKEN`) inline in `.github/workflows/ci.yml` for the same reason.

**When adding new test files**: always ensure `tests/conftest.py` is loaded first (pytest does this automatically for files under `tests/`). Never import from `src` at the module level of a test file without the conftest guard.

**When adding new required settings fields**: update `tests/conftest.py` `os.environ.setdefault()` block AND the CI typecheck job env vars, or both will break.

## Testing

- **Unit tests** (`tests/test_translator.py`, `tests/test_app.py`): Mock the `WebClient` via `unittest.mock.patch("src.main.WebClient")`. Use the custom `MockWebClient` class (not `AsyncMock`) — it has real async methods so `lifespan` can call `await aclose()`. Chat completions tests also mock `openai.AsyncOpenAI` via `patch("src.main.openai.AsyncOpenAI")` using `MockAsyncOpenAI` which supports `chat.completions.create()` and `close()`.
- **Dual-patch pattern**: Every unit test patches both `WebClient` and `AsyncOpenAI` via the `_patches()` helper — both must be patched even when testing only one route, because the FastAPI `lifespan` creates both clients at startup.
- **Integration tests** (`tests/integration/`): Use real credentials. Skip automatically when env vars are test defaults. The skip guard reads from the already-instantiated `settings` singleton, not raw env vars.
- **OpenAI SDK tests** (`tests/integration/test_openai_sdk.py`): Wire the OpenAI client through `TestClient` via `http_client=client` with `base_url="http://testserver/v1"`. Covers models, streaming, tool calls, and thinking variants.
- **Parallel tool call tests** (`tests/integration/test_parallel_tool_calls.py`, `test_orchestrator_tool_calls.py`): Verify proxy doesn't drop/truncate parallel tool call deltas; stress test with 8-way parallel subagent roundtrips.
- Integration tests need real env vars **exported before pytest starts** (not just in `.env`) because `os.environ.setdefault` in the root conftest won't overwrite pre-existing vars.
- **Async generator mock pattern**: Tests use `yield  # noqa: F841` after `raise` to make async functions into generators. This is intentional — do not remove the `# noqa` comments.
- **Settings-dependent tests** (e.g. `TestStreamEmptyRetry`): patch `src.main.settings` and set ALL attributes the route handler reads (not just the one under test), or you'll get `AttributeError`.
- **Captured kwargs pattern**: Tests asserting upstream params use `captured: dict = {}` closure in the mock handler, then assert on `captured["model"]`, `captured["extra_body"]`, etc.

## Conventions

- **Python 3.12+**, `ruff` for linting, `pyright` (standard mode) for type checking.
- **Line length**: 120 chars.
- **Ruff rules**: `E, F, W, I, UP` only. No docstring or naming convention enforcement.
- **Env management**: conda (`conda activate open-webui-openai-compatible`).
- **All config in `pyproject.toml`** — no standalone ruff.toml, pyrightconfig.json, etc.
- **`tui.py` is not covered by CI** — `ruff check` and `pyright` both only target `src/` and `tests/`.
- **No pytest config** — no `[tool.pytest.ini_options]` in pyproject.toml; test paths must be specified explicitly. Running bare `pytest` discovers integration tests that need real credentials.

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

The `AsyncOpenAI` client's `base_url` is set to `{open_webui_url}/api` (not `/v1`) so the SDK's `/chat/completions` path maps correctly to `/api/chat/completions`.

Two HTTP clients coexist: `WebClient` (raw httpx, models only) and `openai.AsyncOpenAI` (chat completions only). The split exists because models use Open WebUI's non-OpenAI JSON shape requiring raw httpx, while chat uses the OpenAI SDK's SSE streaming.

## CI pipeline

GitHub Actions (`.github/workflows/ci.yml`): 4 jobs, all on Python 3.12.

1. **lint** — `ruff check src/ tests/` (installs only `ruff`, not full package)
2. **typecheck** — `pyright src/` (with dummy env vars — required because settings singleton fires at import time)
3. **unit-tests** — `pytest tests/test_translator.py tests/test_app.py -v`
4. **integration-tests** — runs after lint+typecheck+unit pass; two-layer skip: job-level `if` blocks fork PRs, shell-level null check handles missing secrets

`lint`, `typecheck`, and `unit-tests` run in parallel. `integration-tests` fans in after all three pass.

No Docker build/push, no pip caching, no deploy step — purely quality gates.

## Constraints

- No hardcoded URLs anywhere in `src/` — all from `settings`.
- Error responses must use OpenAI JSON format: `{"error": {"message", "type", "code"}}`.
- Never expose `OPEN_WEBUI_URL` or `USER_TOKEN` values in error messages or logs.
- Only 3 routes exist. No `/v1/models/{id}`, no embeddings, no CORS middleware.
- `max_retries=0` on the `AsyncOpenAI` client — proxy handles retries itself. Do not raise above 0 or retries will double-fire.
- `_SSE_HEADERS` must always be present on `StreamingResponse` — stripping them breaks streaming behind reverse proxies.
- Do not remove the `# type: ignore[call-arg]` on `Settings()` (`settings.py:36`) — Pyright can't see pydantic-settings env injection.
- Do not remove the `# type: ignore[union-attr]` on `main.py:293` or `# type: ignore[arg-type]` on `main.py:329` — flow-guaranteed non-None, Pyright can't prove it.
- `tests/test_app.py` has `# noqa: F841` on async generator lines (8 instances) — intentional, do not remove.
- Adding a new upstream parameter requires updating `_SDK_KNOWN_PARAMS` (`main.py:47`) or it silently routes to `extra_body`.
- Haiku models must never receive `type="adaptive"` thinking config — `_supports_adaptive()` in `translator.py:52` enforces this.
