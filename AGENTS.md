# AGENTS.md

**Commit:** `ff82e2b` · **Branch:** `main`

FastAPI proxy exposing OpenAI- and Anthropic-compatible endpoints in front of an Open WebUI instance, authenticating with a user's browser JWT instead of an API key. Includes a headless-browser sidecar that obtains and auto-renews that JWT.

Nested guides: [`src/proxy/openai/AGENTS.md`](src/proxy/openai/AGENTS.md) · [`src/proxy/anthropic/AGENTS.md`](src/proxy/anthropic/AGENTS.md) · [`tests/AGENTS.md`](tests/AGENTS.md)

## Quick reference

```sh
# Install (pip, not conda — there is no environment.yml)
pip install ".[dev]"

# Lint + typecheck (both target src/ and tests/ only; tui.py and playwright_login.py are NOT covered)
ruff check src/ tests/
pyright src/

# Unit tests — the 4 files CI runs
python -m pytest tests/test_openai_translator.py tests/test_openai_routes.py tests/test_anthropic_translator.py tests/test_anthropic_routes.py -v

# Unit tests NOT in CI (test_playwright_login.py needs playwright + a real browser)
python -m pytest tests/test_auth.py tests/test_refresh_trigger.py tests/test_playwright_login.py -v

# Integration tests — credentials must be EXPORTED BEFORE pytest starts
OPEN_WEBUI_URL=https://your-instance.example.com USER_TOKEN=<jwt> python -m pytest tests/integration/ -v

# Run server
cp .env.example .env   # then edit
uvicorn src.main:app --port 8000

# Docker (builds Dockerfile.playwright, NOT Dockerfile) — auto-login + refresh timer
docker compose up -d
docker compose exec proxy python /app/playwright_login.py   # force a token refresh
```

## Structure

```
open-webui-openai-compatible/
├── src/
│   ├── settings.py       # Pydantic Settings singleton — instantiated at import time (:38)
│   ├── auth.py           # Token provider: token file → USER_TOKEN env → RuntimeError
│   ├── client.py         # httpx wrapper — used ONLY for /v1/models
│   ├── errors.py         # classify/log upstream errors + OpenAI-format error builder
│   ├── main.py           # 69 lines: _TokenAuth hook, lifespan clients, /health
│   ├── models.py         # Back-compat re-export → src.proxy.openai.models
│   ├── translator.py     # Back-compat re-export → src.proxy.openai.translator
│   └── proxy/
│       ├── openai/       # GET /v1/models, POST /v1/chat/completions  → see its AGENTS.md
│       └── anthropic/    # POST /v1/messages                          → see its AGENTS.md
├── tests/                # 7 unit + 5 integration                     → see its AGENTS.md
├── playwright_login.py   # Login sidecar: headless Chromium → JWT → token file
├── entrypoint.sh         # Container startup: login if needed, refresh loop, exec uvicorn
├── tui.py                # Standalone Textual client; talks to the PROXY. No src/ imports
├── Dockerfile            # Minimal image — NOT used by compose
├── Dockerfile.playwright # What compose builds (Chromium + entrypoint.sh)
├── docs/adr/             # 0001 — rationale for the proxy/ sub-package split
└── .github/workflows/ci.yml
```

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| OpenAI routes / thinking variants / Bedrock scrubbing | `src/proxy/openai/` | Has its own AGENTS.md |
| Anthropic route / streaming translation | `src/proxy/anthropic/` | Has its own AGENTS.md |
| Tests, mocks, fixtures | `tests/` | Has its own AGENTS.md |
| Token source / expiry checks | `src/auth.py` | `get_current_token()` is the only token entry point |
| Browser login / token file format | `playwright_login.py` | UA Shibboleth + Duo selectors; writes `{token, expires_at, retrieved_at}` |
| Container startup / refresh timer | `entrypoint.sh` | `REFRESH_INTERVAL_SECONDS` (default 7200) |
| Add env var | `src/settings.py` | Then update `tests/conftest.py` defaults AND the CI typecheck env |
| Change upstream URL for models | `src/client.py` | `/api/models` only; chat goes through the openai SDK |
| Shared error handling | `src/errors.py` | Both OpenAI and Anthropic formats |
| Design rationale for the layout | `docs/adr/0001-anthropic-api-frontend-via-translation.md` | No status field; implemented |

## Architecture

| Module | Role |
|--------|------|
| `src/settings.py` | Settings singleton — **instantiated at import time** (`:38`) |
| `src/auth.py` | Stateless token provider. No caching, no lock, all sync |
| `src/client.py` | httpx wrapper for `GET /api/models` only. Accepts an injected client |
| `src/errors.py` | `classify_upstream_error`, `log_upstream_error`, `create_openai_error` |
| `src/main.py` | `_TokenAuth(httpx.Auth)` (`:29`), `_lifespan` (`:37`), `app` (`:62`), `/health` (`:67`) |
| `src/models.py`, `src/translator.py` | Back-compat re-export shims (`# noqa: F401`) |
| `tui.py` | Textual chat client. Reads `.env` directly via `load_dotenv`; `PROXY_URL` (default `http://localhost:8000`); sends no auth header |

Dependency direction: `main → routes → {auth, errors, settings, translator} → models`. Two edges worth knowing: `anthropic/routes.py` imports the **private** `_split_body_for_sdk` from `openai/routes.py`, and `errors.py` depends on `proxy/openai/models.py` for the shared error shapes.

### Routes — exactly 4

| Route | Defined at | Upstream |
|-------|-----------|----------|
| `GET /health` | `main.py:67` | — |
| `GET /v1/models` | `proxy/openai/routes.py:236` | `GET /api/models` (via `WebClient`) |
| `POST /v1/chat/completions` | `proxy/openai/routes.py:269` | `POST /api/chat/completions` (via openai SDK) |
| `POST /v1/messages` | `proxy/anthropic/routes.py:55` | `POST /api/chat/completions` (via openai SDK) |

No `/v1/models/{id}`, no embeddings, no CORS middleware. The proxy deliberately avoids Open WebUI's own `/v1/*` paths — those require an `sk-` API key, not a JWT. `AsyncOpenAI.base_url` is `{open_webui_url}/api` so the SDK's `/chat/completions` lands on `/api/chat/completions`.

Two HTTP clients coexist: `WebClient` (raw httpx, models only — Open WebUI returns a non-OpenAI JSON shape there) and `openai.AsyncOpenAI` (chat only — for SSE streaming). Both are created in `_lifespan` and stored on `app.state`.

## Authentication & token refresh

The JWT is never baked into a client. `_TokenAuth.auth_flow` (`main.py:30-33`) calls `get_current_token()` and overwrites the `Authorization` header on **every outgoing request**, so a refreshed token takes effect on the next request with no restart and no client rebuild. `AsyncOpenAI` gets a placeholder `api_key="proxy-auth-via-hook"` (`main.py:52`) that the hook replaces.

Token resolution order (`auth.py:83`): token file (`TOKEN_FILE`, default `~/.config/open-webui-proxy/token.json`) → `USER_TOKEN` env → `RuntimeError`. The file is JSON `{token, expires_at, retrieved_at}`, mode 0600; the proxy reads only `token` and decodes `exp` from the JWT itself.

Refresh paths:
1. **Startup** (`entrypoint.sh`): if no usable token and `UA_NETID`/`UA_NETID_PASSWORD` are set, run `playwright_login.py` before serving.
2. **Timer** (`entrypoint.sh:86-91`): a background shell loop every `REFRESH_INTERVAL_SECONDS` (default 7200). A loop, not cron, so the container needs no root.
3. **On 401** (`proxy/openai/routes.py`): if the token is a decodable-but-expired JWT, or the error body carries a code in `_TOKEN_REJECTION_CODES`, spawn the sidecar (flock-guarded, detached) and return **503**. The client retries; the proxy does not retry the request itself.

`is_token_expired_or_invalid` returns `False` for a *malformed* token on purpose (`auth.py:37-41`) — a browser re-login cannot repair a garbled token, so it must not trigger a refresh.

**The Anthropic route has no 401-refresh logic.** It passes 401s through. Since both routes share one token file and one auth hook, a refresh triggered by an OpenAI-route request does benefit `/v1/messages` — but an Anthropic-originated 401 never starts one.

## Environment variables

| Variable | Read by | Default |
|----------|---------|---------|
| `OPEN_WEBUI_URL` | `settings.py:13` (required), `playwright_login.py`, `entrypoint.sh` | — |
| `USER_TOKEN` | `auth.py:79`, `settings.py:14` (now **optional**) | — |
| `TOKEN_FILE` | `auth.py:20`, sidecar, compose | `~/.config/open-webui-proxy/token.json` (`/data/token.json` in Docker) |
| `UA_NETID` / `UA_NETID_PASSWORD` | `settings.py:15-16`, `playwright_login.py` | — |
| `BROWSER_PROFILE_DIR` | sidecar, `entrypoint.sh`, compose | `/data/browser-profile` |
| `REFRESH_INTERVAL_SECONDS` | `entrypoint.sh` only | `7200` |
| `PLAYWRIGHT_HEADLESS` | `playwright_login.py` | `true` |
| `PORT` / `REQUEST_TIMEOUT` / `STREAM_EMPTY_RETRY_MAX` / `LOG_LEVEL` | `settings.py:17-20` | `8000` / `300` (10-3600) / `3` (0-10) / `INFO` |
| `PROXY_URL` | `tui.py` only | `http://localhost:8000` |

`.env.example` documents 8 of these; `TOKEN_FILE`, `BROWSER_PROFILE_DIR`, `REFRESH_INTERVAL_SECONDS`, `PLAYWRIGHT_HEADLESS`, and `PROXY_URL` are not in it.

## Critical gotcha: settings singleton

`src/settings.py:38` runs `settings = Settings()` at **module level**:

- Importing *any* `src` module triggers validation. Missing `OPEN_WEBUI_URL` crashes the import with `ValidationError`. `USER_TOKEN` is now optional, so its absence does not.
- Tests survive via `os.environ.setdefault()` at the top of `tests/conftest.py`, which runs before any `src` import during collection.
- CI's typecheck job injects dummy env vars inline in `.github/workflows/ci.yml` for the same reason.
- The `# type: ignore[call-arg]` on that line is required — Pyright cannot see pydantic-settings' env injection. Do not remove it.

When adding a **required** settings field, update `tests/conftest.py` and the CI typecheck env, or both break.

## Request flow

1. Client sends an OpenAI-compatible request (or an Anthropic one, which `translate_request()` converts first).
2. `rewrite_chat_body()` runs **seven ordered passes**: strip unsupported fields → incompatible-`thinking` reconciliation → incompatible-`reasoning_effort` stripping → Bedrock tool scrubbing → stream-usage injection → `chat_id` injection → `session_id` stripping. Details in `src/proxy/openai/AGENTS.md`.
3. `resolve_thinking_model()` strips any `:extended`/`:adaptive` suffix and returns a thinking config (only for Anthropic models that support it; the suffix is stripped either way).
4. `apply_thinking_params()` injects `thinking` and raises `max_tokens` to a sufficient floor.
5. `_split_body_for_sdk()` splits fields into SDK kwargs vs `extra_body` using `_SDK_KNOWN_PARAMS`.
6. `AsyncOpenAI.chat.completions.create()` forwards upstream.
7. Chat responses pass through (or are translated back for Anthropic); only the model list is reshaped.

## Streaming

Chunks arrive as parsed `ChatCompletionChunk` objects and are re-serialized with `model_dump_json(exclude_unset=True)`, terminated by `data: [DONE]`.

**First-chunk pre-read**: `__anext__()` is called before `StreamingResponse` is returned, so an immediate upstream rejection becomes a proper HTTP error instead of a broken stream.

**Empty-stream retry**: total attempts = `1 + settings.stream_empty_retry_max`, backoff `min(1 << attempt, 120)`. **4xx is never retried.** On exhaustion via `StopAsyncIteration`, a synthetic `finish_reason="stop"` chunk is returned as a valid 200 SSE stream.

**Finish-reason guard**: if no chunk carried a non-null `finish_reason`, one is synthesized so clients don't hang.

Mid-stream failures (headers already sent) emit `data: {"error": ...}` then `data: [DONE]`. A mid-stream 401 is *not* refresh-eligible — only the two pre-stream windows are.

Error mapping: `APIStatusError` → preserve status · `APITimeoutError` → 504 · `APIConnectionError` → 502 · anything else → 502, logged with the exception type.

## Claude thinking variants

Models from an Anthropic family (`claude`, `fable`, `mythos` in the ID) get virtual variants appended to `/v1/models`: `:extended` (`thinking.type=enabled`, 32k budget / 16k Haiku) and `:adaptive` (`thinking.type=adaptive`, only for Opus/Sonnet >= 4.6 and the `fable`/`mythos` families — Claude 4.5 and earlier reject adaptive upstream). The suffix is stripped before forwarding, including when the variant is refused.

Two distinct capability gates, and they are NOT the same line:

| Gate | Meaning | Applies to |
| --- | --- | --- |
| `_supports_adaptive` | model ACCEPTS `type="adaptive"` | Opus/Sonnet >= 4.6, `fable`, `mythos` |
| `_requires_adaptive` | model REJECTS `type="enabled"` | Opus/Sonnet >= **4.7**, `fable`, `mythos` |

Claude 4.6 accepts BOTH modes; Claude 4.7+ accepts ONLY adaptive, failing enabled thinking with `400 "thinking.type.enabled" is not supported for this model. Use "thinking.type.adaptive" and "output_config.effort"`. Verified against genai.arizona.edu: `claude-4-6-opus`/`claude-4-6-sonnet` -> 200, `claude-5-opus` -> 400. Consequently `:extended` on an adaptive-only family resolves to the ADAPTIVE config, so the variant stays usable instead of guaranteeing a 400.

A client-supplied `thinking` param is stripped for non-Anthropic models. It is Anthropic-only, and Open WebUI forwards unknown top-level params verbatim, so leaving it in produces upstream `400 unknown_parameter: 'thinking'`. On adaptive-only Anthropic models a client-supplied `type="enabled"` is coerced to `{"type": "adaptive"}` for the same reason.

`reasoning_effort` is passed through untouched for every model EXCEPT the `gpt-5.6` line, where ALL reasoning and verbosity controls (`reasoning_effort`, `reasoning`, `effort`, `verbosity`, `textVerbosity`, `thinking`) are stripped. That family is served via **Bedrock Converse**, which accepts none of them: `reasoning_effort` is remapped upstream onto Bedrock's Anthropic-only `thinking` param and fails with `400 unknown_parameter: 'thinking'` even though this proxy never sent `thinking`; `effort`/`textVerbosity` reach Bedrock verbatim and 400; `verbosity` trips `litellm.UnsupportedParamsError`. Verified: all 400 on gpt-5.6-sol/terra/luna, all 200 on claude-5-opus and gpt-oss-120b.

**Consequence:** gpt-5.6 runs at its DEFAULT reasoning effort on this gateway and the depth cannot be steered. A client asking for `xhigh` gets default, and the request succeeds — the only signal is a WARNING in the proxy log. This is an upstream defect workaround; delete the pass once it is fixed.

## Conventions

- **Python 3.12+**, ruff (lint), pyright (standard mode).
- **Line length 120.** Ruff rules: `E, F, W, I, UP` only.
- **All tool config lives in `pyproject.toml`** — no `ruff.toml`, no `pyrightconfig.json`, no `environment.yml`.
- Dependencies are managed with **pip** (`pip install ".[dev]"`). There is no conda env file despite older docs claiming one.
- `pyproject.toml` **does** define `[tool.pytest.ini_options]`, registering an `integration` marker — which is never actually applied (tests use a skipif instead).
- `tui.py` and `playwright_login.py` are outside CI's lint/typecheck scope.

## CI pipeline

`.github/workflows/ci.yml` — 4 jobs on Python 3.12. `lint` (`ruff check src/ tests/`), `typecheck` (`pyright src/`, dummy env vars), and `unit-tests` run in parallel; `integration-tests` fans in after all three. Integration is double-guarded: a job-level `if` blocks fork PRs, and a shell null check exits 0 when secrets are absent. No build, push, cache, or deploy.

**Known gap:** the `unit-tests` job runs only the original four test files. `test_auth.py`, `test_refresh_trigger.py`, and `test_playwright_login.py` never run in CI.

## Constraints

- No hardcoded URLs in `src/` — everything comes from `settings`.
- Error responses use OpenAI format `{"error": {"message", "type", "code"}}`; the Anthropic route uses Anthropic's format.
- Never put `USER_TOKEN` in a log or error message. Token values are currently never logged. **Note:** `src/client.py:39` does log the upstream base URL at DEBUG.
- `max_retries=0` on `AsyncOpenAI` (`main.py:55`) — the proxy owns retries. Raising it double-fires them.
- `_SSE_HEADERS` must stay on every `StreamingResponse` or streaming breaks behind nginx/Caddy. It is defined **twice** (`proxy/openai/routes.py:34`, `proxy/anthropic/routes.py:28`) — keep them in sync.
- Do not remove these suppressions: `settings.py:38` `[call-arg]`; `proxy/openai/routes.py:333` `[union-attr]` and `:388` `[arg-type]`; `proxy/anthropic/routes.py:102` and `:141` `[union-attr]`.
- Adding an upstream parameter requires adding it to `_SDK_KNOWN_PARAMS` (`proxy/openai/routes.py:41`) or it silently routes to `extra_body`.
- Docker: compose builds `Dockerfile.playwright`. `docker compose down -v` deletes the volume holding the token **and** the browser profile, forcing a full interactive Duo login next start.

## Known issues

| Issue | Location |
|-------|----------|
| `/v1/messages` never calls `resolve_thinking_model`, so an `:extended`/`:adaptive` suffix reaches upstream as a literal model ID and 404s. The OpenAI route handles it; the Anthropic route does not | `proxy/anthropic/routes.py:68-72` |
| `translate_request` maps Anthropic `output_config.effort` to a bare top-level `effort` key, which is not a documented upstream param (adaptive depth is set via `output_config.effort`). Suspect — verify against LiteLLM's Bedrock adapter before relying on it | `proxy/anthropic/translator.py:233` |
| Sidecar fallback path is misspelled with a hyphen (`playwright-login.py`); the real file uses an underscore, so the fallback can never resolve | `proxy/openai/routes.py:139` |
| `_MAX_ERROR_LOG_CHARS` defined but never used | `src/client.py:10` |
| `is_token_expired` duplicates `is_token_expired_or_invalid`; only tests import it | `src/auth.py:54` |
| The 6 Anthropic SSE event classes are declared but unused — the translator emits plain dicts. `AnthropicRequest` and the request-side block models are likewise unused (`translate_request` works on raw dicts) | `proxy/anthropic/models.py:213-241` |
| `docs/architecture.{dot,svg,png}` predate the Anthropic frontend: no `/v1/messages`, client labeled "OpenAI Client". Embedded in README | `docs/` |
| Stale bytecode for deleted tests (`test_app`, `test_translator`) | `tests/__pycache__/` |
