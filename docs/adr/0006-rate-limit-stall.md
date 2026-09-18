# Stall-and-retry on upstream rate limits

The gateway throttles each end user with two request tiers, both reported
as HTTP **400** with a ``Rate limit exceeded`` detail string — not as 429,
with no `Retry-After` and no `RateLimit-*` headers (a raw 429 is accepted
on the detection path too). The exact window-reset time appears only inside
the detail string. Verified 2026-09-18 against genai.arizona.edu (Open
WebUI 0.9.6 fronting a LiteLLM gateway):

- **End-user tier**: 20 requests per 60 s, shared across all chat endpoints
  and models, keyed on the JWT identity. Its detail carries the exact
  window-reset time, so the proxy needs no window model for this tier — it
  sleeps until the advertised reset:

```
{"detail": "Rate limit exceeded for end_user: ... Current limit: 20,
Remaining: 0. Limit resets at: 2026-09-18 15:54:15 UTC"}
```

- **Burst tier**: 10 requests per 60 s, shared across all chat endpoints
  and models. Its detail carries **no** reset timestamp — only the limit
  and unit, and the window boundary is not observable:

```
{"detail": "Rate limit exceeded: 10 requests per minute. Please wait before trying again."}
```

  How the proxy models that window (rolling 60 s, from the process's own
  admission history) is decided in
  [ADR-0007](0007-slot-aware-stall.md).

The burst tier (10/min) is the binding constraint; the end-user tier
(20/min) is the ceiling. Effective single-user throughput: ~10 req/min.
Before this ADR a rate-limited request terminated the client request with
that 400, which clients read as "my request is broken", and immediate
client retries were
pointless because the window would not reopen for up to a minute.

## Decision

**The proxy stalls instead of terminating.** On any of the three pre-header
seams — non-streaming create, streaming create, first-chunk pre-read — on
either frontend, a rate-limited upstream response is swallowed, the proxy
sleeps until the advertised reset, and retries, keeping the client request
open.

- **Detection** is transport-neutral and lives in
  `src/open_webui/rate_limit.py` (`is_rate_limit`): a 429 is a rate limit on
  its face; a 400 is one only when its `detail` string carries
  "Rate limit exceeded" (case-insensitive). A bare 400 is a client bug and
  must never stall. The body-sniffing mirrors the precedent set by
  `should_refresh()` for 401s.
- **The budget is finite and per request**: `RATE_LIMIT_STALL_MAX_SECONDS`
  (default 300; 0 disables stalling and restores the old fail-fast
  behaviour). The clock starts on first detection — a request that never
  hits the limit is unstalled, and the usual per-attempt `REQUEST_TIMEOUT`
  applies as before.
- **Each sleep targets the most precise horizon available**: when the detail
  carries a reset timestamp (end-user tier), sleep until ``reset − 0.5 s``;
  when it does not (burst tier), sleep until the next free slot computed by
  ``SlidingWindowTracker`` from this process's admission history (oldest
  admission + 60 s). The 0.5 s margin makes one extra rejected request the
  price of clock skew; a rejection consumes no slot and re-advertises the
  state.
- **401 keeps priority.** A 401 with positive token-fault evidence still
  goes to the refresh path; a stale token is never burned on stalls.
- **The stall budget is separate from the empty-stream budget**
  (ADR-0003). Rate-limit sleeps never consume `stream_empty_retry_max`
  attempts, and exhaustion surfaces the rate-limit response, not the
  synthetic empty stream. "4xx is never retried" is thereby scoped to
  non-rate-limit 4xx.
- **Exhaustion is a 429 with `Retry-After`** in the frontend's own wire
  format (OpenAI `rate_limit_error`, Anthropic `rate_limit_error`), with
  the gateway's own detail text carried in the message. 429 is the one
  status every client knows how to back off on; the proxy earned the right
  to re-map after swallowing the gateway's non-standard 400. The
  `Retry-After` value is the seconds until the last advertised reset, or —
  for the reset-less burst tier — until the next free slot (ADR-0007).
- **Stalling is independent per request.** No shared gate, no token bucket.
  Concurrent stalled requests each fire at the next advertised reset; the
  losers of the 20-slot race re-stall to the freshly advertised reset. The
  60 s window is coarse enough that this self-stabilizing loop needs no
  coordination, and a shared gate would add process-wide mutable state the
  refresh protocol also has to live with.
- **Mid-stream is out of scope.** A rate limit is charged per request start,
  so it arrives in the pre-header window; once the first byte is out the
  status is committed (ADR-0003), and a mid-stream retry would re-emit a
  whole response the client has already half-consumed. Mid-stream failures
  keep their error-event behaviour.
- A client that disconnects mid-stall cancels its handler task; the sleep
  is interrupted and nothing leaks.

## Consequences

- A client request can now take up to `RATE_LIMIT_STALL_MAX_SECONDS` longer
  than a single upstream attempt. The stall is transparent while it lasts:
  no response bytes have been sent, so there is nothing to un-send.
- Exhaustion answers with a 429 + `Retry-After` where clients used to see a
  400. The message carries the gateway's own words, so an operator reading
  the client log sees the upstream's detail.
- New environment variable `RATE_LIMIT_STALL_MAX_SECONDS` (settings,
  conftest defaults, CI typecheck env, `.env.example`).
- Two new glossary terms: *rate limit* and *stall* (CONTEXT.md).
- The dated entry in `docs/upstream-compatibility.md` records the
  400-not-429 quirk and its removal trigger (the gateway starting to emit a
  real 429 with `Retry-After`).
- `max_retries=0` on the SDK client is unchanged — the proxy owns every
  retry.
