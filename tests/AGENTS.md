# tests/ — read before writing or editing any test

## Files

| File | Tests | Asserts | Targets |
|---|---|---|---|
| `test_openai_translator.py` | 93 | 148 | `src.translator` / `src.models` shims → `proxy.openai` |
| `test_anthropic_translator.py` | 55 | 129 | `proxy.anthropic.translator`, incl. `StreamingState` |
| `test_openai_routes.py` | 36 | 88 | `/v1/models`, `/v1/chat/completions` |
| `test_anthropic_routes.py` | 25 | 75 | `/v1/messages` |
| `test_capabilities.py` | 29 | 40 | `src.open_webui.capabilities` |
| `test_auth.py` | 20 | 21 | `src.auth` token store |
| `test_refresh_trigger.py` | 17 | 25 | 401 → refresh, single-flight, atomic write |
| `test_rate_limit.py` | 38 | 94 | `src.open_webui.rate_limit` detection/stall budget + both route seams (stall → 429 + `Retry-After`) |
| `test_playwright_login.py` | 4 | 10 | the sidecar, with a REAL headless Chromium |

Plus `fakes.py` (shared fakes), `conftest.py`, and 5 files in `tests/integration/`.

CI runs `pytest tests/ --ignore=tests/integration --ignore=tests/test_playwright_login.py`, so **every unit file above runs in CI except `test_playwright_login.py`** (it needs a real browser). The former gap where `test_auth` and `test_refresh_trigger` never ran is closed.

## fakes.py — the shared fake module

Do not hand-roll client fakes in a test file. Import them:

| Export | Purpose |
|---|---|
| `fake_clients(openai_handler=, models_payload=, models_source=)` | Builds an `UpstreamClients` for injection. `models_source` may be an async callable (it may raise, to simulate upstream failure) |
| `completion(...)` | A real `ChatCompletion`; supports `tool_calls`, `thinking_blocks` |
| `chunk(...)` | A real `ChatCompletionChunk`; supports `content`, `finish_reason`, `reasoning_content`, `tool_calls` |
| `usage_chunk(...)` | A trailing usage-only chunk (empty `choices`) — for late-usage tests |
| `models_client(source)` | `httpx.AsyncClient` over `MockTransport`; no network |
| `FakeOpenAI` | Stands in for `openai.AsyncOpenAI`; real `async close()` |
| `parse_sse_events(text)` | `[{"event": ..., "data": ...}]` for Anthropic SSE |
| `sse_data_lines(text)` | Raw `data:` payloads for OpenAI SSE |
| `status_error(code, ...)` | An `openai.APIStatusError` |
| `DUMMY_REQUEST` | A throwaway `httpx.Request` |

Hand-rolled classes, not `AsyncMock`: the lifespan awaits real `close()`/`aclose()` on teardown.

## Route test pattern — injection, NOT namespace patching

The old dual-patch of `src.main.WebClient` + `src.main.openai.AsyncOpenAI` is **gone**, along with `WebClient` itself. Tests now inject through the composition seam:

```python
app = create_app(clients=fake_clients(openai_handler=handler))
with TestClient(app) as tc:
    ...
```

The three route-test files keep a local `_patches()` / `_AppFactory` shim so the existing test bodies read unchanged: `_patches(openai_handler=h)` returns a factory twice, and `p_wc.build()` produces the app. `test_refresh_trigger.py` uses `_app_with(handler)` directly.

**Do not reintroduce `patch("src.main....")` for clients.** There is nothing to patch; pass fakes in.

Still-valid patch targets:
- `patch("src.proxy.openai.routes.settings", mock)` — retry tests. Set EVERY attribute the code path reads with real values (`stream_empty_retry_max`, `log_level`); an unset attribute stays a bare `MagicMock` and the retry arithmetic raises `TypeError`.
- `patch("src.proxy.anthropic.routes.settings", mock)` — same, for the Anthropic empty-stream test.
- `patch("src.proxy.openai.routes.request_refresh")` / `patch("src.proxy.anthropic.routes.request_refresh")` — assert a refresh was or was not requested.
- `patch("src.auth.REFRESH_LOCK_PATH", tmp_path / "l.lock")` and `patch("subprocess.Popen", fake)` — single-flight tests. A `Popen` fake **must** return an object with `.wait()`; `request_refresh` hands it to the reaper thread.

## Mock streams must be async generators

A handler returns an async generator. The unreachable `yield  # noqa: F841` after a `raise` or `return` is what makes the function one — the SDK streaming path requires an async iterator. **10 occurrences: 8 in `test_openai_routes.py`, 2 in `test_anthropic_routes.py`.** Do not remove them.

```python
async def handler(**kwargs):
    async def gen():
        yield chunk(content="hi")
        yield chunk(finish_reason="stop")
    return gen()
```

Asserting upstream params: a `captured: dict = {}` closure plus `captured.update(kwargs)`, then assert on `captured["model"]` and `captured.get("extra_body")`.

## conftest.py

- Module level sets seven env vars via `os.environ.setdefault` BEFORE any `src` import: `OPEN_WEBUI_URL`, `USER_TOKEN`, `PORT`, `REQUEST_TIMEOUT`, `LOG_LEVEL`, `TOKEN_FILE`, `RATE_LIMIT_STALL_MAX_SECONDS`. `src/settings.py:39` runs `settings = Settings()` at import time, so a missing var crashes collection.
- `TOKEN_FILE` is `/nonexistent/open-webui-proxy-test/token.json` — an impossible path, so `get_current_token()` falls back to `USER_TOKEN` instead of reading the developer's real token.
- The autouse `mock_settings` fixture uses `monkeypatch.setenv` (hard override per test); module level uses `setdefault` (fills gaps only). Different layers, different semantics.
- `TEST_DEFAULT_URL` / `TEST_DEFAULT_TOKEN` are a CONTRACT: `integration/conftest.py::_is_real_instance()` compares the imported settings singleton against them. Integration env vars must therefore be **exported before pytest starts** — `.env` or mid-session changes do not work.

## test_auth.py

Every `TestGetCurrentToken` test calls `importlib.reload(src.auth)` before `get_current_token()`, because `DEFAULT_TOKEN_FILE` is computed at import. `setup_method`/`teardown_method` save and restore the real `TOKEN_FILE` env var so the developer's path never leaks across tests. The duplicate `is_token_expired` no longer exists — its class now targets `is_token_expired_or_invalid`.

## test_refresh_trigger.py

Pins the whole refresh protocol:
- 401 + positive token-fault evidence → `request_refresh` called once, **503** returned. The client retries; the proxy never retries.
- `TestSingleFlightOwnership` — the proxy must NOT hold the flock across the spawn (a probe inside the fake `Popen` asserts the lock is free), and must NOT unlink the lock path.
- `TestAtomicTokenWrite` — mode 0600, no temp files left behind, and the inode changes on rewrite (proving `os.replace`, so no reader sees partial JSON).
- `TestSidecarReaping` — a spawned sidecar is waited on, because uvicorn is PID 1 and does not reap.

## test_playwright_login.py

HIDDEN DEPENDENCY: imports the root `playwright_login.py` and drives a REAL headless Chromium against a local `http.server` on `127.0.0.1:0`. Requires playwright plus an installed browser. A bare `pytest tests/` will attempt it.

## Integration files

All five set `pytestmark = [skip_without_real_instance, pytest.mark.flaky(reruns=2, reruns_delay=5)]` (via pytest-rerunfailures). The `openai_client` fixture is REDEFINED per file in `test_openai_sdk`, `test_parallel_tool_calls`, `test_orchestrator_tool_calls` rather than shared. `pyproject.toml` registers an `integration` marker that is never applied (tests use the skipif). Zero `async def test_` functions; `pytest-asyncio` is an unused dev dep.

## Where to add a test

| Task | File |
|---|---|
| Rewrite passes, SDK split | `test_openai_translator.py` (via the shim) |
| Model family / adaptive gates / reasoning controls | `test_capabilities.py` |
| Thinking variants, model-list translation | `test_openai_translator.py` |
| `/v1/chat/completions`, `/v1/models`, SSE, retry | `test_openai_routes.py` |
| Anthropic request/response translation, stream lifecycle | `test_anthropic_translator.py` |
| `/v1/messages` behaviour, suffix stripping, 401 refresh | `test_anthropic_routes.py` |
| Token file/env fallback, expiry | `test_auth.py` |
| 401 → refresh, lock ownership, atomic write, reaping | `test_refresh_trigger.py` |
| Rate-limit detection, stall budget, 429 exhaustion | `test_rate_limit.py` |
| Sidecar browser login (needs a browser) | `test_playwright_login.py` |
| Real-instance behaviour | `tests/integration/` |

## Gotchas

- Patching `src.main.WebClient`: the attribute does not exist. Inject via `create_app(clients=...)`.
- A `subprocess.Popen` fake returning `None`: `request_refresh` passes it to the reaper, which calls `.wait()`. Return a `MagicMock()`.
- Removing a `yield  # noqa: F841`: the handler stops being an async generator and the streaming path breaks.
- Patching `...routes.settings` with a bare mock and setting only the attr under test: `TypeError` in retry arithmetic.
- Asserting `message_delta` comes out of `translate_chunk`: it does not. Terminal events come only from `finalize()`.
- Expecting parallel tool-call blocks to stream through: they are buffered until `finalize()` so blocks never interleave.
- Setting integration creds in `.env` or mid-session: `_is_real_instance()` sees defaults and every test skips.
- A bare `pytest tests/`: drags in `test_playwright_login` (real browser).
