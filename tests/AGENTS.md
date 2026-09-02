# tests/ - read before writing or editing any test

## Files
- Unit (7): test_openai_translator.py, test_openai_routes.py, test_anthropic_translator.py, test_anthropic_routes.py, test_auth.py, test_refresh_trigger.py, test_playwright_login.py
- Integration (5): tests/integration/{test_e2e,test_openai_sdk,test_anthropic_e2e,test_parallel_tool_calls,test_orchestrator_tool_calls}.py
- CI GAP: the CI unit-tests job runs ONLY the original four (test_openai_translator, test_openai_routes, test_anthropic_translator, test_anthropic_routes). test_auth, test_refresh_trigger, test_playwright_login never run in CI.
- Stale: tests/__pycache__/test_app.*.pyc and test_translator.*.pyc are leftovers from a rename (test_app.py is now test_openai_routes.py). Ignore them.

## conftest.py
- Module level sets SIX env vars via os.environ.setdefault BEFORE any src import: OPEN_WEBUI_URL, USER_TOKEN, PORT, REQUEST_TIMEOUT, LOG_LEVEL, TOKEN_FILE. src/settings.py runs settings = Settings() at import time, so missing vars crash collection.
- TOKEN_FILE = "/nonexistent/open-webui-proxy-test/token.json": an impossible path so get_current_token() falls back to USER_TOKEN instead of reading the developer's real ~/.config token.
- Autouse mock_settings fixture uses monkeypatch.setenv (hard override per test). Module level uses setdefault (fills gaps only). Different layers, different semantics.
- TEST_DEFAULT_URL / TEST_DEFAULT_TOKEN are a CONTRACT: integration/conftest.py _is_real_instance() compares the imported settings singleton against them. Consequence: integration env vars must be EXPORTED before pytest starts. .env or mid-session changes don't work.

## Route test pattern (test_openai_routes.py, test_anthropic_routes.py)
- DUAL-PATCH, always both, even when testing one route:
  patch("src.main.WebClient", return_value=wc)
  patch("src.main.openai.AsyncOpenAI", return_value=oa)
  The lifespan constructs and closes BOTH clients at startup. TestClient(app) runs it.
- _patches() helper (test_openai_routes.py:110, test_anthropic_routes.py:124) returns both patches. Use it.
- Mocks are hand-rolled classes (MockWebClient, MockAsyncOpenAI, MockChat, MockChatCompletions), NOT AsyncMock. Real async close/aclose methods because the lifespan awaits them on teardown.
- A mock stream handler must be an async GENERATOR: the unreachable yield  # noqa: F841 after raise/return makes it one. 9 occurrences (8 in test_openai_routes.py, 1 in test_anthropic_routes.py). The SDK streaming path requires an async iterator. Do not remove.
- Asserting upstream params: captured: dict = {} closure in the handler, captured.update(kwargs), then assert captured["model"], captured.get("extra_body").
- Settings-dependent tests (TestStreamEmptyRetry) patch src.proxy.openai.routes.settings and set EVERY attribute the code path touches with real values (current tests set stream_empty_retry_max and log_level). Unset attrs stay bare MagicMock. Retry arithmetic (attempt <= max_retries, 1 + max_retries) breaks.
- test_refresh_trigger.py adds three patch targets: src.proxy.openai.routes._trigger_refresh, src.proxy.openai.routes.REFRESH_LOCK_PATH, subprocess.Popen.
- Refresh contract this file pins: 401 + positive token-fault evidence → sidecar spawned ONCE, 503 returned immediately. The client retries. The proxy never retries the request.

## test_auth.py
- Every TestGetCurrentToken test calls importlib.reload(src.auth) before get_current_token().
- Why: auth reads TOKEN_FILE/USER_TOKEN from env at call time, and DEFAULT_TOKEN_FILE is computed at import (src/auth.py:16). Reload picks up per-test env and recomputes the default (the default-path test patches Path.home).
- setup_method/teardown_method save and restore the real TOKEN_FILE env var: tests mutate env per test and the developer's actual token file path must not leak across tests.

## test_playwright_login.py
- HIDDEN DEPENDENCY: imports the ROOT playwright_login.py sidecar and drives a REAL headless Chromium against a local http.server on 127.0.0.1:0.
- Requires playwright + an installed browser. Bare pytest tests/ will attempt these.

## Integration files
- All five set: pytestmark = [skip_without_real_instance, pytest.mark.flaky(reruns=2, reruns_delay=5)] (flaky via pytest-rerunfailures).
- The openai_client fixture is REDEFINED PER FILE in test_openai_sdk, test_parallel_tool_calls, test_orchestrator_tool_calls (not shared from integration/conftest.py): OpenAI(base_url="http://testserver/v1", api_key="not-needed", http_client=client) over the shared client fixture.
- pyproject.toml registers an integration marker but it is NEVER APPLIED (tests use the skipif instead). No asyncio config, zero async def test_ functions. pytest-asyncio is an unused dev dep.

## Where to add a test
| Task | File |
|---|---|
| Body rewriting, thinking params, model translation (OpenAI) | test_openai_translator.py |
| /v1/chat/completions and /v1/models behavior, status codes, SSE, retry | test_openai_routes.py |
| Anthropic request/response/streaming translation | test_anthropic_translator.py |
| /v1/messages route behavior | test_anthropic_routes.py |
| Token file/env fallback, expiry checks | test_auth.py |
| 401 → sidecar spawn, lock file, _should_refresh_token | test_refresh_trigger.py |
| Sidecar browser login (needs playwright + browser) | test_playwright_login.py |
| Real-instance basics (health, models, chat, SSE) | tests/integration/test_e2e.py |
| OpenAI SDK compatibility | tests/integration/test_openai_sdk.py |
| /v1/messages against a real instance | tests/integration/test_anthropic_e2e.py |
| Tool calls against a real instance | tests/integration/test_parallel_tool_calls.py, test_orchestrator_tool_calls.py |

## Gotchas
- New route test missing either patch: lifespan builds real clients and the test hangs or fails.
- AsyncMock for the SDK client: lifespan awaits real aclose()/close() on teardown.
- Removing a yield  # noqa: F841: handler stops being an async generator, streaming path breaks.
- Setting integration creds in .env or mid-session: _is_real_instance() sees defaults and every test skips.
- Patching src.proxy.openai.routes.settings with a bare mock and setting only the attr under test: TypeError in retry arithmetic.
- Bare pytest tests/: drags in test_playwright_login (real browser) and the integration conftest import.
- Trusting tests/__pycache__/test_app.*.pyc or test_translator.*.pyc: stale bytecode from a deleted/renamed file.
- Expecting CI to cover a new test: only the original four files run.
