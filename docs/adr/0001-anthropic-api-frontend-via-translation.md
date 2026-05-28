# Add Anthropic Messages API as a second proxy frontend

The proxy currently exposes only OpenAI-compatible routes (`/v1/models`, `/v1/chat/completions`). We're adding a `POST /v1/messages` endpoint that accepts Anthropic Messages API format, translates requests to OpenAI format for Open WebUI, and translates responses back to Anthropic format.

This means the proxy becomes a **multi-format frontend** for a single OpenAI-speaking backend (Open WebUI). Both SDKs (OpenAI and Anthropic) can target the same proxy instance without any upstream changes.

## Considered Options

1. **Anthropic translation layer (chosen)** — Accept Anthropic-format requests, translate bidirectionally. Clients use whichever SDK they prefer.
2. **Require all clients to use OpenAI format** — Simpler, but forces users with Anthropic SDK tooling to switch. Since the upstream models *are* Claude (via Bedrock), many users naturally reach for the Anthropic SDK.
3. **Separate proxy instance for Anthropic** — Avoids shared code but doubles deployment and config burden for no architectural benefit.

## Consequences

- **Source restructure**: Flat `src/` layout becomes `src/proxy/openai/` and `src/proxy/anthropic/` sub-packages. Shared infrastructure (`settings`, `client`, `errors`) stays in `src/` root. This is delivered as a separate PR before the Anthropic feature lands.
- **Translation fidelity**: Upstream (Open WebUI via Bedrock) returns `thinking_blocks` with real Anthropic `signature` values, so multi-turn thinking works at full fidelity. No need to fake or omit signatures.
- **Streaming complexity**: Anthropic SSE format (`message_start`/`content_block_delta`/`message_stop`) is structurally different from OpenAI's flat chunk stream. The translation is stateful — must track content block indices and lifecycle.
- **Maintenance surface**: Every new Anthropic API feature (content block types, parameters) needs a corresponding translation mapping. This is the ongoing cost of the approach.
