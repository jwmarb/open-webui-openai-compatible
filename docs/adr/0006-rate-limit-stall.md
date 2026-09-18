# Stall-and-retry on upstream rate limits

The gateway throttles each end user to a fixed number of requests per window.
Verified 2026-09-18 against genai.arizona.edu: 20 requests per 60 s per end
user, and overage is reported as HTTP **400** with a body of

```
{"detail": "Rate limit exceeded for end_user: ... Current limit: 20,
Remaining: 0. Limit resets at: 2026-09-18 15:54:15 UTC"}
```

— not a 429, with no `Retry-After` and no `RateLimit-*` headers. The exact
window-reset time appears only inside the detail string. Before this ADR a
rate-limited request terminated the client request with that 400, which
clients read as "my request is broken", and immediate client retries were
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
- **Each sleep is `min(max(0, reset − 0.5 s), 1 s, time-to-deadline)`**,
  where 1 s is the fallback when no reset timestamp is parseable. The 0.5 s
  margin makes one extra rejected request the price of clock skew; a
  rejection consumes no slot and re-advertises the reset.
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
  `Retry-After` value is the seconds until the last advertised reset.
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
