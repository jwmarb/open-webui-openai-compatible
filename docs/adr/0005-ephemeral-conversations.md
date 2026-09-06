# Ephemeral Open WebUI conversations

`POST /api/chat/completions` is Open WebUI's internal chat endpoint. It expects
to be driven by its own web UI, which owns a conversation record and a
WebSocket session.

## Decision

Every request is made ephemeral. `chat_id` is set to `local:<uuid4>` and
`session_id` is stripped.

**The `local:` prefix is a persistence decision, not only a crash workaround.**
Open WebUI 0.9.x does require a non-`None` `chat_id` (it calls
`.startswith` on it), but the prefix specifically tells it to skip all database
persistence: no conversation rows, no ownership checks, no history lookup. API
traffic therefore leaves no trace in the user's chat history, which is what an
API client expects. A caller wanting persistence would have to opt in
explicitly, and no route offers that today.

**`session_id` is stripped** because its presence routes the request through
Open WebUI's WebSocket task pool for multi-model fan-out. The proxy holds no
WebSocket connection, so fan-out hangs.

## Consequences

- Proxy traffic is invisible in the Open WebUI interface. This is intended;
  users surprised by it are usually looking for their chat history.
- Conversation state is entirely the client's responsibility, which matches
  both the OpenAI and Anthropic APIs.
- If persistence is ever wanted, it needs a deliberate opt-in and an ownership
  story, not the removal of the prefix.
