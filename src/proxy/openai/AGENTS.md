# src/proxy/openai/

The OpenAI-compatible **frontend**: `GET /v1/models` (routes.py:153) and `POST /v1/chat/completions` (routes.py:185). It owns OpenAI wire format only — request shapes, SSE framing, error bodies, and the thinking-variant surface. **Gateway policy lives in `src/open_webui/`, not here.**

Vocabulary: [`CONTEXT.md`](../../../CONTEXT.md).

## Where to look

| Task | Location |
|------|----------|
| Route handlers, streaming, empty-stream retry, 401 refresh | `routes.py` |
| Thinking-variant generation/resolution, model-list translation | `translator.py` |
| OpenAI error body | `errors.py` |
| Pydantic response shapes | `models.py` |
| The seven rewrite passes, SDK split | `../../open_webui/request_policy.py` |
| What a model accepts | `../../open_webui/capabilities.py` |

## What this package no longer owns

Backend policy moved out. Nothing here re-derives a model family, and no private symbol is imported across a package seam.

| Was here | Now |
|---|---|
| `_SDK_KNOWN_PARAMS` (private) | `open_webui/request_policy.py:29` `SDK_KNOWN_PARAMS` (**public**) |
| `_split_body_for_sdk` (private, imported by the Anthropic route) | `open_webui/request_policy.py:222` `split_body_for_sdk` |
| The 7 rewrite passes + `rewrite_chat_body` | `open_webui/request_policy.py` |
| 7 model-family predicates (`_is_claude_model`, `_supports_adaptive`, `_requires_adaptive`, `_adaptive_capability`, `_extract_version`, `_normalize_model_id`, `_is_small_context_claude`) | `open_webui/capabilities.py` → one `capabilities_for()` call |
| `create_openai_error` (lived in `src/errors.py`) | `errors.py` in this package |
| `_should_refresh_token`, `_extract_error_code`, `_trigger_refresh`, `_TOKEN_REJECTION_CODES`, `REFRESH_LOCK_PATH`, `PROJECT_ROOT`, `_REFRESH_SCRIPT` | `src/auth.py` (the token store) |

`rewrite_chat_body` and `sanitize_chat_body` are still importable from `translator.py` as re-exports, because `tests/test_openai_translator.py` and the `src/translator.py` shim consume them.

## routes.py

- `_SSE_HEADERS` (`:31`: `Cache-Control: no-cache`, `X-Accel-Buffering: no`) — required on every `StreamingResponse` or SSE breaks behind nginx/Caddy. Duplicated at `../anthropic/routes.py:29`; keep in sync.
- `_RETRY_BACKOFF_CAP` (`:36`, 120s). Duplicated at `../anthropic/routes.py:104`.
- `_upstream_error_response` (`:44`) and `_token_expired_error_response` (`:51`) build OpenAI-format bodies via `errors.create_openai_error`.
- `_refresh_for` (`:62`) is the only token logic left in this file: on a 401 it asks `auth.should_refresh(get_current_token(), exc.body)` and calls `auth.request_refresh()`. Positive evidence only — a bare 401 is not enough. Four call sites: models (`:161`), streaming create (`:238`), streaming first-chunk (`:249`), non-streaming (`:311`). Each returns **503**; the client retries, the proxy never retries a 401 itself.
- Streaming retry (`_handle_streaming` `:220`): total attempts `1 + settings.stream_empty_retry_max`, backoff `min(1 << attempt, _RETRY_BACKOFF_CAP)`. 4xx short-circuits and is never retried. The first chunk is pre-read via `__anext__` before `StreamingResponse` returns, so an immediate rejection surfaces as an HTTP error rather than a broken stream. Exhausted `StopAsyncIteration` yields a synthetic `finish_reason="stop"` chunk. Rationale: [ADR-0003](../../../docs/adr/0003-streaming-retry-and-commit-semantics.md).
- `models` (`:154`) reads the injected client directly: `request.app.state.models_client.get("/api/models")` then `raise_for_status()`. There is no `WebClient` wrapper any more, and no `app.state.web_client`.
- `# type: ignore[union-attr]` and `# type: ignore[arg-type]` in the streaming path are flow-guaranteed non-None. Do not remove.

## translator.py

142 lines. Thinking variants and model-list translation only.

- `generate_thinking_variants` (`:48`): appends `:extended` for any Anthropic family, plus `:adaptive` when `capabilities_for(id).supports_adaptive`.
- `resolve_thinking_model` (`:74`): strips the suffix and returns the config. A refused suffix is STILL stripped (returns the base, config `None`) so the request does not 404 upstream on a synthetic model ID. On an adaptive-only family `:extended` resolves to `ADAPTIVE_THINKING_CONFIG` rather than an enabled budget, keeping the suffix usable instead of guaranteeing a 400.
- `apply_thinking_params` (`:110`): injects `thinking` and raises `max_tokens` to a floor (`MIN_MAX_TOKENS_EXTENDED` 64k, `MIN_MAX_TOKENS_EXTENDED_SMALL` 32k for Haiku).
- `translate_models_response` (`:126`): reshapes `GET /api/models` into `/v1/models` and appends variants per model.
- Thinking configs: `EXTENDED_THINKING_CONFIG` (32k budget), `EXTENDED_THINKING_CONFIG_SMALL` (16k, Haiku), `ADAPTIVE_THINKING_CONFIG`.

All family questions go through `capabilities_for()`. This module contains no regex, no version parsing, and no family token set.

## models.py

`OpenAIModel` (`:10`), `OpenAIModelList` (`:19`), `ThinkingConfig` (`:26`), `OpenAIErrorDetail` (`:33`), `OpenAIErrorResponse` (`:41`). The error shapes are consumed by `errors.py` in this package — `src/errors.py` no longer imports them.

## Cross-package contract

The Anthropic frontend imports two **public** symbols from this package: `resolve_thinking_model` and `apply_thinking_params` (`../anthropic/routes.py:22`). Changing either signature breaks that route. It imports nothing private.

## Constraints

- Adding an upstream parameter requires adding it to `SDK_KNOWN_PARAMS` (`../../open_webui/request_policy.py:29`) or it silently routes to `extra_body`.
- This package must not format an Anthropic error, and the backend package must not format an OpenAI one.
- Package tests: `tests/test_openai_translator.py` (95 tests, via the `src.translator` shim), `tests/test_openai_routes.py` (36 tests).
