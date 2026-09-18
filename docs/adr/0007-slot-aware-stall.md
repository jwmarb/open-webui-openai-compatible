# Slot-aware stall for the sliding request tier

ADR-0006 stalls a rate-limited request until the advertised window reset.
That targets the end-user tier (20 requests per 60 s), whose detail carries
the exact reset time. The gateway's second tier does not advertise one — its
detail is

```
{"detail":"Rate limit exceeded: 10 requests per minute. Please wait before trying again."}
```

— so the stall had nothing to target and a 1 s fallback sleep stood in:
stalled requests polled every second for up to a minute and re-collided at
the same instant.

Verified 2026-09-18 against genai.arizona.edu (Open WebUI 0.9.6 + LiteLLM):
the sliding tier is 10 requests per 60 s, per end user. It is global across
models (a burst on model A exhausts the budget for model B), shared by both
chat endpoints, and unaffected by the `user` field in the request body. A
rejected request consumes no slot, and a full burst of 10 is blocked for
~15-60 s and never released early. The sustainable rate is exactly 10/min.

## Decision

**The proxy sleeps to the next free slot of the sliding tier.**
`SlidingWindowTracker` in `src/open_webui/rate_limit.py` (process singleton)
models the gateway's window from this process's own admission history. The
proxy is the sole consumer of the token, so that history is a *conservative*
model of the gateway's window: the next free slot is computed as the oldest
admission plus the window length. The measured behaviour (a full burst is
never released early) is consistent with a window anchored at the first
admission, so rolling only ever sleeps longer than the gateway's true next
admission — and a too-early retry is a free 400, so the conservatism is safe.

- **Admissions are recorded at the pre-header seams** — streaming create,
  first-chunk pre-read, and non-streaming create, on both frontends — via
  `record_upstream_admission()`. A rate-limit rejection at any of those
  seams removes the admission via `record_upstream_rejection()`, so
  rejections consume no slot, matching the gateway.
- **The tier parameters come from the detail string.**
  `rpm_tier_from_detail()` parses the limit and unit out of the sliding
  tier's detail and the tracker adopts them; the observed 10 per 60 s
  stands as the default before any detail has been seen. The end-user
  tier's detail does not match, so it never re-shapes the window model.
- **Each sleep is `oldest_admission + window − now`, fired 0.25 s early**
  (with a 50 ms floor). The early fire costs one extra rejected request,
  and a rejection consumes no slot, so the margin stays small. There is no
  jitter: concurrent stalled requests fire at the same slot and the losers
  re-stall to the next one. Free rejections self-stabilize the loop, as
  ADR-0006's reset loop does, so no shared gate is added.
- **With no history the 1 s fallback stands in** — a fresh process has
  nothing to model. The first accepted requests build the history that
  makes later stalls exact.
- **The end-user tier is untouched.** When the detail carries a reset
  timestamp the stall targets it, per ADR-0006.
- **Budget clamping is unchanged.** Slot-aware sleeps are clamped to
  `RATE_LIMIT_STALL_MAX_SECONDS` exactly as reset-aware ones are; exhaustion
  still surfaces the rate-limit response as a 429.
- **Exhaustion `Retry-After` now reflects the next slot.** When no reset
  timestamp was advertised, `retry_after()` answers the seconds until the
  tracker's next slot instead of a flat 1.

## Consequences

- A stalled request sleeps the shortest correct duration against this
  process's own admissions instead of polling every second. Note the scope:
  this schedules *stalls*, it does not pace traffic — after a burst of
  `limit` requests, the next stalled request waits for the oldest admission
  to expire, and the following burst may fire as soon as the window empties.
- The tracker is new process-wide mutable state, shared by every request.
  It needs no coordination: admissions are appended at the pre-header
  seams, and every stalled request sleeps on the oldest slot, so losers of
  a slot race simply re-stall.
- `Retry-After` on exhaustion is meaningful for the sliding tier, not just
  the end-user tier.
- New public seam: `record_upstream_admission()` /
  `record_upstream_rejection()` at all six pre-header seams (three per
  frontend: streaming create, first-chunk pre-read, non-streaming create).
- **Removal trigger:** if the gateway starts advertising a reset timestamp
  in the sliding tier's detail, the tracker is no longer needed — delete
  `SlidingWindowTracker` and the admission recording, and let ADR-0006's
  reset-targeting path cover both tiers.
