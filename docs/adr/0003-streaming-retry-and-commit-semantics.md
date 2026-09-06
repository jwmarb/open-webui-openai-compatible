# Streaming retry and commit semantics

Open WebUI occasionally returns a stream that yields no chunks. Clients see a
successful response that never produces content, which looks like a hang.

## Decision

**The first chunk is pre-read before the response is returned.** Awaiting
`__anext__()` while we can still change the status code converts an immediate
upstream rejection into a proper HTTP error instead of a 200 that breaks
mid-stream.

**Only the pre-stream window is retried.** Total attempts are
`1 + stream_empty_retry_max` with backoff `min(1 << attempt, 120)`. Once the
first chunk is out, headers are sent and the status is committed; a later
failure can only be reported as an SSE `error` event.

**4xx is never retried.** A client error is deterministic — retrying it wastes
the backoff window and, for a 401, delays the refresh path.

**`max_retries=0` on the SDK client.** The proxy owns retry policy. Leaving the
SDK's own retries enabled would multiply attempts and make the backoff
unpredictable.

**Retrying carries a duplicate-work risk we accept.** An upstream that accepted
the request, began billable work, and emitted nothing is indistinguishable from
one that never started. We retry anyway, because the alternative is a client
that hangs. `stream_empty_retry_max` exists so operators can set this to zero.

**Exhaustion returns a synthetic successful stream.** After the final attempt we
emit a terminating chunk and a normal `[DONE]`. A well-formed empty answer is
easier for clients to handle than an error arriving where content was promised.

**A missing `finish_reason` is synthesized.** Clients wait for it; upstream does
not always send one.

## Consequences

- Transient empty streams recover without client involvement.
- Mid-stream failures degrade to an error event, never a changed status code.
- The retry policy lives in one module and applies to both frontends.
