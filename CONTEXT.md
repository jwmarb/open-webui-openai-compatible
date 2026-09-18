# CONTEXT

Domain language for this proxy. Use these terms exactly; they name the seams.

## The system

A **proxy** that presents two **frontends** — OpenAI-compatible and
Anthropic-compatible — in front of one **backend**, an Open WebUI instance. It
authenticates with a user's browser **JWT** rather than an API key, because
institutional deployments disable API-key generation.

## Terms

**Frontend** — a wire-format adapter. Owns request/response shapes, SSE framing
and error bodies for one external API. Lives in `src/proxy/<format>/`. There are
exactly two: `openai` and `anthropic`.

**Backend** — Open WebUI, reached at `POST /api/chat/completions` and
`GET /api/models`. Its requirements live in `src/open_webui/` and are shared by
both frontends. "Backend" never means a frontend's implementation.

**Gateway** — the backend plus whatever it routes to (LiteLLM, AWS Bedrock
Converse). Use this when a constraint comes from further upstream than Open
WebUI itself, which is most of them.

**Canonical body** — an OpenAI-shaped chat body, after any frontend translation
but before backend policy. Both frontends produce one; `prepare_chat_body()`
consumes it. The Anthropic frontend translates *into* canonical shape first.

**Rewrite pass** — one transformation applied to a canonical body by
`src/open_webui/request_policy.py`. There are eight and their order is
load-bearing.

**Capability** — what the gateway will accept for a given model, inferred from
its ID because `GET /api/models` exposes no metadata. A `ModelCapabilities`
value, produced by `capabilities_for()`.

**Model family** — a group of models sharing a request surface: `claude`,
`fable`, `mythos` are Anthropic families. Detected by a positive allowlist.

**Thinking variant** — a virtual model ID formed by appending `:adaptive`.
Advertised in `/v1/models`, resolved and stripped before the request goes
upstream. A variant is never a real upstream model ID. `:extended` was a second
variant; it was removed and is no longer recognised.

**Stream lifecycle** — the ordered SSE events of one response. On the Anthropic
side `StreamingState` owns it end to end: exactly one `message_start`, blocks
that never interleave, exactly one terminal `message_delta`/`message_stop`.

**Content block** — one Anthropic response segment (text, thinking, tool_use).
Its `index` must equal its position in the final content array, so blocks are
emitted sequentially and never overlap.

**Token store** — `src/auth.py`. Owns the token file, expiry judgement and
renewal requests. The only interface for credentials.

**Sidecar** — `playwright_login.py`. Drives a headless browser through
institutional SSO to obtain a fresh JWT, and owns the single-flight lock for the
duration of a renewal.

**Refresh eligibility** — whether an upstream 401 carries positive evidence the
token is at fault. A bare 401 is not evidence.

**Rate limit** — the gateway's per-end-user request budgets (Open WebUI user
rate limiting), reported as HTTP 400 with a "Rate limit exceeded" detail —
not as 429. Verified at genai.arizona.edu 2026-09-18: a binding tier of 10
requests per rolling 60 s (global across models; detail carries no reset
time), plus a 20-per-60 s tier whose detail carries the exact window-reset
time.

**Stall** — holding a client request open, before any response byte is sent,
while the proxy retries upstream until the rate-limit window resets. A stall is
invisible to the client and bounded by a per-request budget; on exhaustion the
proxy answers 429 with `Retry-After`.
## Deliberately absent

**No "service" or "manager".** Modules are named for what they own:
`request_policy`, `capabilities`, `auth`.

**No request-side or SSE-event Pydantic models on the Anthropic side.**
`translate_request` and `StreamingState` work on raw dicts. Declaring inert
models would imply validation that does not happen.

**No persistence.** Every request is ephemeral by construction — see
[ADR-0005](docs/adr/0005-ephemeral-conversations.md).
