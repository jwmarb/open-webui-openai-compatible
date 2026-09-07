# src/proxy/anthropic/

The Anthropic Messages API **frontend**: `POST /v1/messages` (routes.py:76). Translates Anthropic requests into a **canonical body**, sends them upstream via the OpenAI SDK, and translates responses back. Design rationale: [ADR-0001](../../../docs/adr/0001-anthropic-api-frontend-via-translation.md).

Vocabulary: [`CONTEXT.md`](../../../CONTEXT.md).

## Route flow

`routes.py:76-103`, in this order — the order is load-bearing:

1. `translate_request(raw_body)` → canonical (OpenAI-shaped) body.
2. `resolve_thinking_model()` + `apply_thinking_params()` — strips `:adaptive` and injects the thinking config. **Without this the suffix reached upstream as a literal model ID and 404'd.** `:extended` is no longer recognised anywhere.
3. `prepare_chat_body()` → the seven **rewrite passes** plus the SDK/`extra_body` split, in one call.
4. `_handle_streaming` (`:107`) or `_handle_non_streaming` (`:189`).

## Cross-package imports — all public

- `routes.py:22` — `resolve_thinking_model`, `apply_thinking_params` from `../openai/translator.py`.
- `routes.py:20` — `prepare_chat_body` from `../../open_webui/request_policy.py`.
- `routes.py:18-19` — `classify_upstream_error`, `log_upstream_error` from `src/errors.py`; `get_current_token`, `request_refresh`, `should_refresh` from `src/auth.py`.

No private symbol crosses a package seam. The former `from ..openai.routes import _split_body_for_sdk` is gone.

Still duplicated with `../openai/routes.py`, keep in sync: `_SSE_HEADERS` (`:29` ↔ `:31`), `_RETRY_BACKOFF_CAP` (`:104` ↔ `:36`).

## 401 handling — now symmetric with the OpenAI route

`_refresh_for` (`:37`) returns True when the exception is a 401 **and** `should_refresh(get_current_token(), exc.body)` finds positive token-fault evidence; it then calls `request_refresh()`. `_token_expired_error_response` (`:48`) returns **503** in Anthropic error format. Three call sites: streaming create (`:130`), streaming first-chunk (`:137`), non-streaming (`:203`).

A 401 without token evidence (e.g. `model_access_denied`) still passes through as a 401. The earlier asymmetry — where an Anthropic-only client never triggered a refresh — is fixed.

## translator.py

517 lines. Request/response translation and the stream lifecycle.

- `translate_request` (`:182`): Anthropic → canonical. Default `max_tokens` 4096. `top_k` passthrough. `output_config` maps to `effort` + `response_format` json_schema. system, tools, tool_choice, thinking all mapped.
- `_map_finish_reason` (`:245`): `stop`→`end_turn`, `length`→`max_tokens`, `tool_calls`→`tool_use`, `content_filter`→`end_turn`, unknown→`end_turn`.
- `translate_response` (`:257`): canonical → Anthropic. Upstream `thinking_blocks` keep their real `signature` values, so non-streaming multi-turn thinking works at full fidelity.
- `create_anthropic_error` (`:24`): the Anthropic error body.

### StreamingState (`:313`) — owns the whole stream lifecycle

The class is the single authority on event ordering. The route feeds chunks and calls `finalize()` once; it does **not** infer lifecycle (the old `saw_finish` flag is gone).

Invariants, each pinned by a test:

| Invariant | Method |
|---|---|
| Exactly one `message_start`, always the first event | `translate_chunk` (`:418`) emits it on first call; `finalize` (`:496`) emits it if the stream never started |
| `translate_chunk` NEVER emits terminal events | it records `_stop_reason` instead |
| Exactly one `message_delta` + `message_stop`, only from `finalize()` | `finalize` (`:496`), guarded by `_finalized` |
| `finalize()` is idempotent | second call returns `[]` |
| Usage is absorbed from ANY chunk, including a trailing usage-only chunk | `_absorb_usage` (`:399`) |
| `total_tokens` is NEVER an output-token fallback | Anthropic usage has no such field |
| Content blocks never interleave; a block's `index` equals its position in the final content array | `_flush_tool_blocks` (`:379`) |
| `signature_delta` is emitted when upstream supplies a signature | `translate_chunk` |

**Parallel tool calls are buffered, not streamed through.** Upstream interleaves tool-call fragments by `index`; Anthropic requires sequential, non-overlapping blocks (its SDK appends on `content_block_start` and indexes `content[event.index]` on delta, so an interleaved stream raises). `_buffer_tool_calls` (`:480`) accumulates id/name/arguments per upstream index; `_flush_tool_blocks` (`:379`) emits each as a complete start→delta→stop block at `finalize()`. This trades a little latency on tool calls for wire correctness.

All events are plain dicts; `routes.py` serializes them via `_sse_event` (`:71`).

## models.py

47 lines, four shapes, all used at runtime: `AnthropicUsage`, `AnthropicResponse`, `AnthropicErrorDetail`, `AnthropicErrorResponse` (imported by `translator.py:9-14`).

The 6 SSE event classes, `AnthropicRequest`, and every request-side content-block model are **deleted**. They were declared but unreachable, which implied validation that never happened. `translate_request` and `StreamingState` work on raw dicts by design — see the "Deliberately absent" section of `CONTEXT.md`.

## Where to look

| Task | Where |
|------|-------|
| Add an Anthropic request parameter | `translator.py` `translate_request`. If the key is not in `SDK_KNOWN_PARAMS` (`../../open_webui/request_policy.py:29`) it routes to `extra_body` |
| Add a content block type | Request side: `_translate_content_blocks` (`translator.py:73`). Response side: `translate_response` (`:257`) |
| Change SSE event shape or ordering | `StreamingState` (`translator.py:313`). NOT models.py |
| Change error shape | `create_anthropic_error` (`translator.py:24`) + `_anthropic_error_response` (`routes.py:52`) |
| Streaming lifecycle | `StreamingState` + `_handle_streaming` (`routes.py:107`) |

## Constraints / gotchas

- `# type: ignore[union-attr]` in the streaming path: do not remove. The SDK stream type is a union pyright cannot narrow.
- Translation order is load-bearing: `translate_request` → thinking resolution → `prepare_chat_body`. Do not reorder.
- Never emit a content block while another is open, and never emit an index out of order — Anthropic clients accumulate per open block.
- Every new Anthropic API feature needs a matching translation mapping here. That is the accepted ongoing cost of the translation approach (ADR-0001).
- Package tests: `tests/test_anthropic_translator.py` (54 tests), `tests/test_anthropic_routes.py` (25 tests). Both run in CI.
