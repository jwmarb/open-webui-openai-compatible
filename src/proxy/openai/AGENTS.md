# src/proxy/openai/

The OpenAI-compatible surface: `GET /v1/models` (routes.py:236) and `POST /v1/chat/completions` (routes.py:269). Reads a request, rewrites it (translator.py), splits it for the openai SDK (routes.py), and streams or returns the result. Shared infra (`settings`, `client`, `errors`, `auth`) lives at `src/` root.

## Where to look

| Task | Location |
|------|----------|
| Route handlers, streaming, empty-stream retry, 401 refresh | `routes.py` |
| Body rewriting, thinking-variant generation/resolution | `translator.py` |
| Pydantic response shapes | `models.py` |
| Cross-package consumer (imports helpers below) | `../anthropic/routes.py` |

## Cross-package contracts (breaking these breaks the Anthropic route)

- `_split_body_for_sdk` (routes.py:58) is imported from `../anthropic/routes.py:17`. A private symbol crossing a package boundary. Signature change = Anthropic route break.
- `rewrite_chat_body` is imported from `../anthropic/routes.py:18`.
- `_SSE_HEADERS` (routes.py:34: `Cache-Control: no-cache`, `X-Accel-Buffering: no`) is duplicated at `../anthropic/routes.py:28`. Keep both in sync. Required on every `StreamingResponse` or SSE breaks behind nginx/Caddy.
- `models.py` error shapes (`OpenAIErrorDetail` :33, `OpenAIErrorResponse` :41) are consumed by `src/errors.py`.

## routes.py

- `_SDK_KNOWN_PARAMS` (routes.py:41-48): frozenset of params `AsyncOpenAI.chat.completions.create()` accepts as kwargs. Anything not in the set goes to `extra_body`. Adding an upstream param without adding it here = silent mis-routing to `extra_body`.
- `_split_body_for_sdk` (routes.py:58) partitions the body on that set.
- Streaming retry (routes.py:303-391): total attempts = `1 + settings.stream_empty_retry_max`. Backoff `min(1 << attempt, _RETRY_BACKOFF_CAP)` (routes.py:50, applied at :346 and :359). 4xx short-circuits, never retried (routes.py:342-343, :355-356). First chunk is pre-read via `__anext__` before `StreamingResponse` returns, so an immediate rejection surfaces as an HTTP error, not a broken stream. Exhausted `StopAsyncIteration` yields a synthetic `finish_reason="stop"` chunk (routes.py:367-380).
- `# type: ignore[union-attr]` (routes.py:333) and `# type: ignore[arg-type]` (routes.py:388): flow-guaranteed non-None. Do not remove.

## Token refresh (routes.py)

- Constants: `REFRESH_LOCK_PATH` (:51, `/tmp/openwebui-proxy-refresh.lock`), `PROJECT_ROOT` (:52), `_REFRESH_SCRIPT` (:53, `playwright_login.py`), `_TOKEN_REJECTION_CODES` (:91, `{"invalid_issuer", "invalid_token", "token_expired"}`).
- `_extract_error_code` (:94) pulls the code from upstream JSON. `_should_refresh_token` (:108) = token expired/invalid OR code in `_TOKEN_REJECTION_CODES`. A bare 401 is not enough.
- `_trigger_refresh` (:119): non-blocking `fcntl.flock` + detached `subprocess.Popen` (`start_new_session=True`). Sidecar logs to `/tmp/sidecar.log`.
- Four 401-refresh sites, each lazily imports `get_current_token`: models (:245-256), streaming create (:318-326), streaming first-chunk (:334-341), non-streaming (:412-419). Positive evidence: spawn sidecar, return 503. The client retries. The proxy never retries a 401 request itself.
- KNOWN BUG (routes.py:139): fallback `script_path = "playwright-login.py"` uses a HYPHEN; the real file is `playwright_login.py` (underscore). Dead path, the fallback never resolves.

## translator.py

- `rewrite_chat_body` (:325) is a SIX-pass pipeline in this exact order (order matters):
  1. `_strip_unsupported_fields` (:295) drops `_UNSUPPORTED_FIELDS` (:287: `vector_store_ids`, `file_ids`).
  2. `_strip_incompatible_thinking` (:262): drops a CLIENT-SUPPLIED `thinking` param when the target model is not an Anthropic family. `thinking` is Anthropic-only and OWUI forwards unknown top-level params verbatim, so leaving it in causes upstream `400 unknown_parameter: 'thinking'`. Runs BEFORE `apply_thinking_params`, so the proxy's own suffix-derived injection stays authoritative.
  3. `_scrub_bedrock_tool_fields` (:220): drops legacy `functions`/`function_call`, coerces Bedrock-incompatible `tool_choice`, injects `_DUMMY_TOOL` when history references tools.
  4. `_ensure_stream_usage` (:252): forces `stream_options.include_usage=true` when `stream` is true.
  5. `_inject_chat_id` (:302): sets `chat_id="local:<uuid4>"`. Without it OWUI 0.9.x crashes (`NoneType.startswith`); the `local:` prefix makes OWUI skip DB persistence.
  6. `_strip_session_id` (:314): removes `session_id` to prevent OWUI WebSocket multi-model fan-out (the proxy has no WS connection).
- `sanitize_chat_body` (:336) is an alias of `rewrite_chat_body`.
- Thinking constants: `EXTENDED_THINKING_CONFIG` (:21, 32k budget), `EXTENDED_THINKING_CONFIG_SMALL` (:22, 16k, Haiku), `MIN_MAX_TOKENS_EXTENDED` (:25, 64k), `MIN_MAX_TOKENS_EXTENDED_SMALL` (:26, 32k).
- `resolve_thinking_model` (:145) strips `:extended`/`:adaptive` and returns the config. `apply_thinking_params` (:172) injects `thinking` and raises `max_tokens` to the floor. A refused suffix is STILL stripped (returns `base`, config `None`) so the request does not 404 upstream on a synthetic model ID.
- Model-family detection: `_normalize_model_id` (:56) lowercases and unifies `.`/`_`/`/` to `-`, so `anthropic.claude-x` and `bedrock_claude_x` agree. `_is_claude_model` (:79) is a positive allowlist over `_ANTHROPIC_FAMILY_TOKENS` (:66: `claude`, `fable`, `mythos`) — NOT "not OpenAI", because an unrecognised model must never receive a provider-specific param. OWUI's `GET /api/models` exposes no capability metadata, so ID matching is the only signal. Add new Anthropic families to that frozenset as they ship.
- `_supports_adaptive` (:96) is enforced in BOTH `generate_thinking_variants` (:126) and `resolve_thinking_model` (:159). Adaptive requires an Opus/Sonnet line at version >= 4.6 (`_ADAPTIVE_MIN_VERSION` :43), or a `fable`/`mythos` family. Claude 4.5 and earlier accept only `type="enabled"` and reject adaptive with `400 adaptive thinking is not supported on this model`. `_extract_version` (:51) bounds version digits to `\d{1,2}` so trailing date stamps (`...-4-6-20250514`) are not misread as versions.

## models.py

- `OpenAIModel` (:10), `OpenAIModelList` (:19): `/v1/models` shapes. `translate_models_response` (translator.py:113) appends thinking variants per model.
- `ThinkingConfig` (:26): `type` + optional `budget_tokens`.
