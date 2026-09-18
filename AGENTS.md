# AGENTS.md

FastAPI proxy exposing OpenAI- and Anthropic-compatible endpoints in front of an Open WebUI instance, authenticating with a user's browser JWT instead of an API key. Includes a headless-browser sidecar that obtains and auto-renews that JWT.

Domain language: [`CONTEXT.md`](CONTEXT.md) — read it first. The terms *frontend*, *backend*, *gateway*, *canonical body*, *rewrite pass*, *capability*, *thinking variant*, *stream lifecycle*, *content block*, *token store* and *sidecar* are used precisely throughout.

Nested guides: [`src/proxy/openai/AGENTS.md`](src/proxy/openai/AGENTS.md) · [`src/proxy/anthropic/AGENTS.md`](src/proxy/anthropic/AGENTS.md) · [`tests/AGENTS.md`](tests/AGENTS.md)

## Quick reference

```sh
# Install (pip, not conda — there is no environment.yml)
pip install ".[dev]"

# Lint + typecheck (lint covers the sidecar; pyright covers src/ only)
ruff check src/ tests/ playwright_login.py
pyright src/

# Unit tests — exactly what CI runs
python -m pytest tests/ --ignore=tests/integration --ignore=tests/test_playwright_login.py -v

# Needs playwright + a real browser, never run in CI
python -m pytest tests/test_playwright_login.py -v

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
├── CONTEXT.md            # domain glossary — the seams have names, use them
├── src/
│   ├── settings.py       # Pydantic Settings singleton — instantiated at import time (:38)
│   ├── auth.py           # TOKEN STORE: file/env read, expiry, refresh protocol, atomic write
│   ├── errors.py         # transport-neutral upstream error classification (38 lines)
│   ├── main.py           # create_app / build_upstream_clients / UpstreamClients / app
│   ├── models.py         # Back-compat re-export → src.proxy.openai.models
│   ├── translator.py     # Back-compat re-export → src.proxy.openai.translator
│   ├── open_webui/       # BACKEND POLICY, shared by both frontends
│   │   ├── capabilities.py    # capabilities_for() — one capability lookup
│   │   ├── rate_limit.py      # is_rate_limit / RateLimitStall — stall budget (ADR-0006)
│   │   └── request_policy.py  # 8 rewrite passes, SDK split
│   └── proxy/
│       ├── openai/       # GET /v1/models, POST /v1/chat/completions  → see its AGENTS.md
│       └── anthropic/    # POST /v1/messages                         → see its AGENTS.md
├── tests/                # 8 unit files + fakes.py + 5 integration      → see its AGENTS.md
├── playwright_login.py   # Sidecar: headless Chromium → JWT → token file. Owns the refresh lock
├── entrypoint.sh         # Container startup: login if needed, refresh loop, exec uvicorn
├── tui.py                # Standalone Textual client; talks to the PROXY. No src/ imports
├── Dockerfile            # Minimal image — NOT used by compose
├── Dockerfile.playwright # What compose builds (Chromium + entrypoint.sh)
├── docs/adr/             # 0001 frontend split · 0002 token renewal · 0003 streaming retry
│                         # 0004 capability inference · 0005 ephemeral conversations
│                         # 0006 rate-limit stall
├── docs/upstream-compatibility.md   # dated gateway-defect workarounds + removal triggers
└── .github/workflows/ci.yml
```

`src/client.py` and its `WebClient` wrapper are **deleted**. The models route uses the injected `httpx.AsyncClient` directly.

## Where to look

| Task | Location | Notes |
|------|----------|-------|
| Domain vocabulary | `CONTEXT.md` | Names the seams; use these terms in code and docs |
| Gateway request rules (all 7 passes) | `src/open_webui/request_policy.py` | Shared by both frontends |
| Upstream rate-limit stall | `src/open_webui/rate_limit.py` | Detection + per-request stall budget; 429 + `Retry-After` on exhaustion (ADR-0006) |
| What a model accepts | `src/open_webui/capabilities.py` | `capabilities_for()`; rules are empirical, see ADR-0004 |
| OpenAI routes / thinking variants | `src/proxy/openai/` | Has its own AGENTS.md |
| Anthropic route / streaming translation | `src/proxy/anthropic/` | Has its own AGENTS.md |
| Tests, fakes, fixtures | `tests/` | Has its own AGENTS.md |
| Token source / expiry / refresh | `src/auth.py` | The only interface for credentials |
| Browser login / lock ownership | `playwright_login.py` | Sidecar owns the single-flight lock |
| App wiring, client injection | `src/main.py` | `create_app(config, clients)` |
| Container startup / refresh timer | `entrypoint.sh` | `REFRESH_INTERVAL_SECONDS` (default 7200) |
| Add env var | `src/settings.py` | Then update `tests/conftest.py` defaults AND the CI typecheck env |
| Shared error classification | `src/errors.py` | Wire formats live in each frontend |
| A gateway rejects a param | `docs/upstream-compatibility.md` | Dated table + removal trigger, not an ADR |

## Architecture

| Module | Role |
|--------|------|
| `src/settings.py` | Settings singleton — **instantiated at import time** (`:38`) |
| `src/auth.py` | Token store and refresh protocol. Public: `get_current_token` (`:127`), `should_refresh` (`:160`), `request_refresh` (`:178`), `write_token_file` (`:87`), `get_token_expiry`, `is_token_expired_or_invalid`, `get_token_file_path`, `extract_error_code` |
| `src/errors.py` | `classify_upstream_error`, `log_upstream_error`. Transport-neutral — imports no frontend |
| `src/open_webui/rate_limit.py` | `is_rate_limit` (`:48`), `RateLimitStall` (`:82`) — stall budget, ADR-0006 |
| `src/open_webui/capabilities.py` | `capabilities_for()` (`:88`) → frozen `ModelCapabilities` (`:49`) |
| `src/open_webui/request_policy.py` | `rewrite_chat_body` (`:208`), `split_body_for_sdk` (`:222`), `prepare_chat_body` (`:234`), `SDK_KNOWN_PARAMS` (`:29`) |
| `src/main.py` | `UpstreamClients` (`:44`), `build_upstream_clients` (`:53`), `create_app` (`:71`), `app = create_app()` (`:98`), `_TokenAuth` (`:35`) |
| `src/proxy/openai/errors.py` | `create_openai_error` — OpenAI wire format only |
| `src/models.py`, `src/translator.py` | Back-compat re-export shims (`# noqa: F401`). Consumed only by `tests/test_openai_translator.py` |
| `tui.py` | Textual chat client. Reads `.env` directly; `PROXY_URL` (default `http://localhost:8000`); sends no auth header. Imports nothing from `src/` — it is an external consumer of the proxy interface, and that independence is deliberate |

**Dependency direction:** `main → routes → {auth, errors, settings, open_webui} → models`. Both frontends depend on `open_webui`; `open_webui` depends on neither. Nothing in `src/` root imports a frontend — `errors.py` no longer reaches into `proxy/openai/models.py`, and no private symbol crosses a package seam.

### Routes — exactly 4

| Route | Defined at | Upstream |
|-------|-----------|----------|
| `GET /health` | `main.py:91` | — |
| `GET /v1/models` | `proxy/openai/routes.py:153` | `GET /api/models` (injected httpx client) |
| `POST /v1/chat/completions` | `proxy/openai/routes.py:185` | `POST /api/chat/completions` (openai SDK) |
| `POST /v1/messages` | `proxy/anthropic/routes.py:76` | `POST /api/chat/completions` (openai SDK) |

No `/v1/models/{id}`, no embeddings, no CORS middleware. The proxy deliberately avoids Open WebUI's own `/v1/*` paths — those require an `sk-` API key, not a JWT. `AsyncOpenAI.base_url` is `{open_webui_url}/api` so the SDK's `/chat/completions` lands on `/api/chat/completions`.

Two clients coexist in `UpstreamClients`: `models` (raw httpx — Open WebUI returns a non-OpenAI JSON shape there) and `chat` (`openai.AsyncOpenAI`, for SSE streaming). Both are built by `build_upstream_clients()` and closed by `UpstreamClients.aclose()`.

## Composition and injection

`create_app(config=None, clients=None)` is the seam. Passing `clients` skips `build_upstream_clients`, so tests inject fakes without patching module namespaces; passing `config` injects a `Settings` instance. `app = create_app()` is the production default for uvicorn.

## Authentication & token refresh

Authoritative: [ADR-0002](docs/adr/0002-token-renewal-protocol.md).

The JWT is never baked into a client. `_TokenAuth.auth_flow` (`main.py:35`) calls `get_current_token()` and overwrites the `Authorization` header on **every outgoing request**, so a refreshed token takes effect on the next request with no restart. `AsyncOpenAI` gets a placeholder `api_key="proxy-auth-via-hook"` that the hook replaces.

Token resolution order (`auth.py:127`): token file (`TOKEN_FILE`, default `~/.config/open-webui-proxy/token.json`) → `USER_TOKEN` env → `RuntimeError`. The file is JSON `{token, expires_at, retrieved_at}`, mode 0600; the proxy reads only `token` and decodes `exp` from the JWT itself.

Refresh paths:
1. **Startup** (`entrypoint.sh`): if no usable token and `UA_NETID`/`UA_NETID_PASSWORD` are set, run `playwright_login.py` before serving.
2. **Timer** (`entrypoint.sh`): a background shell loop every `REFRESH_INTERVAL_SECONDS` (default 7200). A loop, not cron, so the container needs no root.
3. **On 401** (both frontends): `_refresh_for(exc)` asks `should_refresh(get_current_token(), exc.body)`; on positive evidence it calls `request_refresh()` and returns **503**. The client retries; the proxy never retries the request itself.

**Single-flight ownership belongs to the sidecar, not the caller.** `request_refresh()` spawns the sidecar and returns; the sidecar acquires `flock` on `REFRESH_LOCK_PATH` (`auth.py:34`) for its whole run, and a losing sidecar exits as a no-op. The proxy must never hold that lock across the spawn — doing so made the child's own acquire fail, so a 401 could produce **no refresh at all**. The lock file is never unlinked: a fresh inode lets a later caller acquire immediately and defeats mutual exclusion.

`write_token_file` (`auth.py:87`) replaces the file atomically — mode-0600 temp file in the destination directory, fsync, `os.replace`. A plain write let a concurrent reader see partial JSON and silently fall back to an expired `USER_TOKEN`.

Spawned sidecars are reaped by a daemon thread (`auth.py` `_reap`) because uvicorn runs as PID 1 and does not reap; `docker-compose.yaml` also sets `init: true`.

`is_token_expired_or_invalid` returns `False` for a *malformed* token on purpose. A browser login would in fact overwrite it, but treating local corruption as grounds for renewal turns every request into a refresh storm.

## Environment variables

| Variable | Read by | Default |
|----------|---------|---------|
| `OPEN_WEBUI_URL` | `settings.py:13` (required), `playwright_login.py`, `entrypoint.sh` | — |
| `USER_TOKEN` | `auth.py`, `settings.py:14` (optional) | — |
| `TOKEN_FILE` | `auth.py`, sidecar, compose | `~/.config/open-webui-proxy/token.json` (`/data/token.json` in Docker) |
| `UA_NETID` / `UA_NETID_PASSWORD` | `settings.py:15-16`, `playwright_login.py` | — |
| `BROWSER_PROFILE_DIR` | sidecar, `entrypoint.sh`, compose | `/data/browser-profile` |
| `REFRESH_INTERVAL_SECONDS` | `entrypoint.sh` only | `7200` |
| `PLAYWRIGHT_HEADLESS` | `playwright_login.py` | `true` |
| `PORT` / `REQUEST_TIMEOUT` / `STREAM_EMPTY_RETRY_MAX` / `LOG_LEVEL` | `settings.py:17-20` | `8000` / `300` (10-3600) / `3` (0-10) / `INFO` |
| `RATE_LIMIT_STALL_MAX_SECONDS` | `settings.py` (optional) | `300` (0–3600; 0 disables stalling) |
| `PROXY_URL` | `tui.py` only | `http://localhost:8000` |

## Critical gotcha: settings singleton

`src/settings.py:38` runs `settings = Settings()` at **module level**:

- Importing *any* `src` module triggers validation. Missing `OPEN_WEBUI_URL` crashes the import with `ValidationError`. `USER_TOKEN` is optional, so its absence does not.
- Tests survive via `os.environ.setdefault()` at the top of `tests/conftest.py`, which runs before any `src` import during collection.
- CI's typecheck job injects dummy env vars inline in `.github/workflows/ci.yml` for the same reason.
- The `# type: ignore[call-arg]` on that line is required — Pyright cannot see pydantic-settings' env injection. Do not remove it.

`create_app(config=...)` now accepts an injected `Settings`, so new code should prefer that over reaching for the global. The import-time singleton remains for uvicorn's `src.main:app`.

When adding a **required** settings field, update `tests/conftest.py` and the CI typecheck env, or both break.

## Request flow

1. Client sends an OpenAI-compatible request, or an Anthropic one which `translate_request()` converts into a **canonical body** first.
2. The frontend resolves any **thinking variant** — `resolve_thinking_model()` strips `:adaptive` and returns a config; `apply_thinking_params()` injects `thinking` and raises `max_tokens` to a floor. **Both** frontends do this.
3. `prepare_chat_body()` applies the eight **rewrite passes** and splits the result into SDK kwargs vs `extra_body` on `SDK_KNOWN_PARAMS`.
4. `AsyncOpenAI.chat.completions.create()` forwards upstream.
5. Chat responses pass through, or are translated back for Anthropic; only the model list is reshaped.

The eight passes run in this exact order and the order is load-bearing: strip unsupported fields → reconcile incompatible `thinking` → strip incompatible reasoning controls → strip incompatible `output_config.effort` → scrub Bedrock tool fields → inject stream usage → inject `chat_id` → strip `session_id`. Each receives a `ModelCapabilities` value rather than re-deriving the model family. Details in `src/proxy/openai/AGENTS.md`.

## Streaming

Authoritative: [ADR-0003](docs/adr/0003-streaming-retry-and-commit-semantics.md).

Chunks arrive as parsed `ChatCompletionChunk` objects. The OpenAI route re-serializes them with `model_dump_json(exclude_unset=True)` and terminates with `data: [DONE]`; the Anthropic route translates them through `StreamingState`.

**First-chunk pre-read**: `__anext__()` is called before `StreamingResponse` is returned, so an immediate upstream rejection becomes a proper HTTP error instead of a broken stream.

**Empty-stream retry**: total attempts = `1 + settings.stream_empty_retry_max`, backoff `min(1 << attempt, 120)`. **4xx is never retried** — with one documented exception: the rate-limit class of 400 stalls within `RATE_LIMIT_STALL_MAX_SECONDS` and then answers 429 + `Retry-After` (ADR-0006). On exhaustion via `StopAsyncIteration` a synthetic successful stream is returned.

**Finish-reason guard** (OpenAI route): if no chunk carried a non-null `finish_reason`, one is synthesized so clients don't hang.

Mid-stream failures (headers already sent) emit an error event then terminate. A mid-stream 401 is *not* refresh-eligible — only the pre-stream windows are.

Error mapping: `APIStatusError` → preserve status · `APITimeoutError` → 504 · `APIConnectionError` → 502 · anything else → 502.

## Claude thinking variant

Models that accept adaptive thinking get ONE virtual variant appended to `/v1/models`: `:adaptive`. The suffix is stripped before forwarding by **both** frontends. `:extended` was removed — it is no longer generated or recognised, so `model:extended` now reaches upstream verbatim and 400s.

Three distinct capability gates, and they are NOT the same line:

| Gate | Meaning | Applies to | Used by |
| --- | --- | --- | --- |
| `supports_adaptive` | model ACCEPTS `type="adaptive"` | Opus/Sonnet >= 4.6, `fable`, `mythos` | variant generation + suffix resolution |
| `requires_adaptive` | model REJECTS `type="enabled"` | Opus/Sonnet >= **4.7**, `fable`, `mythos` | coercing a CLIENT-supplied `thinking` |
| `accepts_effort_config` | model ACCEPTS `output_config.effort` | any non-Anthropic, or Claude >= **4.6** | stripping `output_config.effort` |

Claude 4.6 accepts BOTH thinking modes; 4.7+ accepts ONLY adaptive. `requires_adaptive` is still load-bearing without `:extended`, because a client may send `thinking.type="enabled"` itself.

`accepts_effort_config` lines up with neither: Claude 4.5 and earlier 400 on `output_config.effort` outright, 4.6 accepts it and silently IGNORES it, and only 5.x acts on it. The floor is 4.6 because accepting-and-ignoring is harmless while rejecting is not — so a 4.6 model both permits `type="enabled"` AND takes the effort config. Verified 2026-09-06 against genai.arizona.edu. Do not collapse these three. Rationale: [ADR-0004](docs/adr/0004-model-capability-inference.md).

A client-supplied `thinking` param is stripped for non-Anthropic models (it is Anthropic-only and Open WebUI forwards unknown top-level params verbatim, producing `400 unknown_parameter: 'thinking'`). On adaptive-only families a client-supplied `type="enabled"` is coerced to `{"type": "adaptive"}`.

`reasoning_effort` passes through untouched except on the `gpt-5.6` line, where ALL reasoning and verbosity controls are stripped because that family is served via Bedrock Converse and accepts none of them. **Consequence:** gpt-5.6 runs at its default reasoning effort and the depth cannot be steered; a client asking for `xhigh` gets default and the request succeeds. Full verification table and removal trigger: [`docs/upstream-compatibility.md`](docs/upstream-compatibility.md).

## Conventions

- **Python 3.12+**, ruff (lint), pyright (standard mode).
- **Line length 120.** Ruff rules: `E, F, W, I, UP` only.
- **All tool config lives in `pyproject.toml`** — no `ruff.toml`, no `pyrightconfig.json`, no `environment.yml`.
- Dependencies are managed with **pip** (`pip install ".[dev]"`).
- `pyproject.toml` defines `[tool.pytest.ini_options]`, registering an `integration` marker which is never actually applied (tests use a skipif instead).
- `tui.py` is outside pyright's scope. `playwright_login.py` is linted but not typechecked.

## CI pipeline

`.github/workflows/ci.yml` — 4 jobs on Python 3.12. `lint` (`ruff check src/ tests/ playwright_login.py`), `typecheck` (`pyright src/`, dummy env vars), and `unit-tests` run in parallel; `integration-tests` fans in after all three. Integration is double-guarded: a job-level `if` blocks fork PRs, and a shell null check exits 0 when secrets are absent. No build, push, cache, or deploy.

`unit-tests` runs `pytest tests/ --ignore=tests/integration --ignore=tests/test_playwright_login.py`, so every unit file is covered except the one needing a real browser. The former gap where `test_auth`, `test_refresh_trigger` and `test_capabilities` never ran in CI is closed.

## Constraints

- No hardcoded URLs in `src/` — everything comes from `settings`.
- Error responses use each frontend's own format. The backend module must never format a wire-format error, or the coupling it removed comes back.
- Never put `USER_TOKEN` in a log or error message. Token values are never logged.
- `max_retries=0` on `AsyncOpenAI` (`main.py`) — the proxy owns retries. Raising it double-fires them.
- `_SSE_HEADERS` must stay on every `StreamingResponse` or streaming breaks behind nginx/Caddy. It is defined **twice** (`proxy/openai/routes.py:31`, `proxy/anthropic/routes.py:29`), as is `_RETRY_BACKOFF_CAP` (`:36` / `:104`) — keep them in sync.
- Do not remove these suppressions: `settings.py:38` `[call-arg]`; the `[union-attr]` and `[arg-type]` ignores in both routes files.
- Adding an upstream parameter requires adding it to `SDK_KNOWN_PARAMS` (`open_webui/request_policy.py:29`) or it silently routes to `extra_body`.
- Docker: compose builds `Dockerfile.playwright`. `docker compose down -v` deletes the volume holding the token **and** the browser profile, forcing a full interactive Duo login next start.

## Known issues

| Issue | Location |
|-------|----------|
| Structured output is broken upstream for Claude: the gateway rewrites `response_format` into `output_config.format`, which Bedrock rejects with `Extra inputs are not permitted`. Reproduces on BOTH frontends, so it is not a translation defect. Not worked around — see [`docs/upstream-compatibility.md`](docs/upstream-compatibility.md) | gateway |
| `docs/architecture.{dot,svg,png}` predate the Anthropic frontend and the backend package: no `/v1/messages`, client labeled "OpenAI Client". Embedded in README | `docs/` |
| Streaming thinking blocks carry a signature only when upstream sends one. Open WebUI does not always emit `thinking_blocks` mid-stream, so multi-turn replay fidelity is upstream-dependent | `proxy/anthropic/translator.py` |
| `pytest-asyncio` is an unused dev dependency; there are zero `async def test_` functions | `pyproject.toml` |
