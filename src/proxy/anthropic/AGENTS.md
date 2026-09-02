# src/proxy/anthropic/: Anthropic Messages API frontend

Read before editing this package. Design rationale: [`docs/adr/0001-anthropic-api-frontend-via-translation.md`](../../docs/adr/0001-anthropic-api-frontend-via-translation.md) (authoritative for why this package exists and its accepted maintenance cost).

Translates Anthropic Messages API requests to OpenAI chat format, sends them upstream via the OpenAI SDK (`POST /api/chat/completions`), translates responses back. This package is the only Anthropic-speaking frontend.

## Route

- Single route: `routes.py:55` `POST /v1/messages`.
- Flow at `routes.py:68-69`: `translate_request()` (this package) runs **before** `rewrite_chat_body()` (the OpenAI package's rewriter). Order matters: the body must already be in OpenAI shape when the rewriter's passes run.
- Streaming: `_handle_streaming` (`routes.py:82`). Non-streaming: `_handle_non_streaming` (`routes.py:166`). SSE framing via `_sse_event` (`routes.py:51`).
- Empty-stream retry mirrors the OpenAI route: attempts = `1 + settings.stream_empty_retry_max`, backoff `min(1 << attempt, _RETRY_BACKOFF_CAP)`, 4xx never retried; exhaustion on `StopAsyncIteration` returns a synthetic `end_turn` stream (`routes.py:115-128`).

## Not self-contained: cross-package imports

- `routes.py:17` imports the **private** `_split_body_for_sdk` from `../openai/routes.py`.
- `routes.py:18` imports `rewrite_chat_body` from `../openai/translator.py`.
- `routes.py:28` `_SSE_HEADERS` and `routes.py:79` `_RETRY_BACKOFF_CAP` are **duplicates** of `../openai/routes.py:34` and `:50`. Keep in sync. The SSE headers must stay on every `StreamingResponse` or streaming breaks behind the proxy.

## Critical asymmetry: NO 401 token-refresh here

The OpenAI package treats an upstream 401 as possible token failure: it checks `is_token_expired_or_invalid` / rejection codes, spawns the login sidecar, and returns 503 (openai/routes.py:108-152, :319-325, :335-341, :413-419).

This package does none of that. A 401 is passed through as an Anthropic-format error at `routes.py:97-98` (generic), `routes.py:104-105` (4xx branch), and `routes.py:180-181` (non-streaming).

Consequence: a client that only uses `/v1/messages` never triggers a token refresh when the JWT expires. Mitigation: both routes share one token file and one auth hook, so a refresh initiated by any OpenAI-route request also fixes this route. This is a known gap, not an oversight: an Anthropic-originated 401 never initiates a refresh.

## translator.py

- `translate_request` (:182): Anthropic to OpenAI. Default `max_tokens` 4096 (:187). `top_k` passthrough (:218). `output_config` maps to `effort` + `response_format` json_schema (:228-240). system, tools, tool_choice, thinking all mapped.
- `_map_finish_reason` (:245): stop to end_turn, length to max_tokens, tool_calls to tool_use, content_filter to end_turn (unknown to end_turn).
- `translate_response` (:257): OpenAI to Anthropic. Upstream `thinking_blocks` are kept with **real Anthropic `signature` values** (:265-271). Multi-turn thinking works at full fidelity. Nothing is faked or omitted (per the ADR).
- `StreamingState` (:313): stateful OpenAI-chunk to Anthropic-SSE translator. Anthropic SSE (`message_start` / `content_block_*` / `message_delta` / `message_stop`) is structurally different from OpenAI's flat chunk stream, so block indices and lifecycle are tracked. Key methods: `translate_chunk` (:369), `finalize` (:456, emits closing `end_turn` events if no `finish_reason` was seen), private block helpers (:326-367).
- `reasoning_content` deltas map to `thinking` blocks (:390-403). `content` to text blocks (:405-417). `tool_calls` to `tool_use` with args accumulated per tool index (:419-442).
- All events are emitted as plain dicts; routes.py serializes them.

## models.py

- Request/response/error Pydantic shapes. Runtime uses only `AnthropicResponse`, `AnthropicUsage`, `AnthropicErrorResponse`, `AnthropicErrorDetail` (imported by translator.py:9-14).
- **The 6 SSE event classes (:213-241, `MessageStartEvent` through `MessageStopEvent`) are declared but UNUSED at runtime.** The translator emits plain dicts. Editing them changes nothing.
- `AnthropicRequest` (:128) and the request-side block models (:14-84) are likewise unused at runtime. `translate_request` operates on raw dicts.

## Where to look

| Task | Where |
|------|-------|
| Add an Anthropic request parameter | `translator.py` `translate_request`. If the key is not in `_SDK_KNOWN_PARAMS` (openai/routes.py:41), `_split_body_for_sdk` routes it to `extra_body` |
| Add a content block type | Request side: `_translate_content_blocks` (translator.py:73). Response side: `translate_response` (translator.py:277+) |
| Change SSE event shape | `StreamingState.translate_chunk` (translator.py:369). NOT models.py |
| Change error shape | `create_anthropic_error` (translator.py:24) + `_anthropic_error_response` (routes.py:36, type map at :39-44) |
| Streaming lifecycle | `StreamingState` (translator.py:313-466) + `_handle_streaming` (routes.py:82-163) |

## Constraints / gotchas

- `# type: ignore[union-attr]` at `routes.py:102` and `:141`: do not remove. The SDK stream type is a union and pyright cannot narrow it.
- `_SSE_HEADERS` / `_RETRY_BACKOFF_CAP` duplicated with `../openai/routes.py`: keep in sync (or consolidate both copies).
- Translation order is load-bearing: `translate_request` then `rewrite_chat_body`. Do not reorder.
- Every new Anthropic API feature (content block type, parameter) needs a matching translation mapping here. That is the accepted ongoing cost of the translation approach (ADR).
- No 401 refresh on this route (above). 401 clients get a 401 Anthropic-format error, not a 503.
- Package tests: `tests/test_anthropic_translator.py`, `tests/test_anthropic_routes.py` (both run in CI unit-tests).
