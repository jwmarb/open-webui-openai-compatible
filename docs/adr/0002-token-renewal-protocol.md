# Token renewal protocol

The proxy authenticates to Open WebUI with a user's browser JWT rather than an
`sk-` API key, because institutional deployments disable API-key generation.
That token expires, so renewal is part of normal operation rather than an
exceptional path.

## Decision

`src/auth.py` owns the whole protocol behind three entry points:
`get_current_token()`, `should_refresh(token, body)` and `request_refresh()`.
Callers never compose those steps themselves.

**Token source order** is token file, then `USER_TOKEN`, then `RuntimeError`.
The file is the preferred source because it is the only one a renewal can
update; the env var is a bootstrap fallback.

**Refresh eligibility requires positive evidence.** A bare upstream 401 is
ambiguous — revoked model access and rate policy produce one too — so a refresh
happens only when the stored token is a decodable JWT past its `exp`, or the
error body carries `invalid_issuer`, `invalid_token` or `token_expired`.
Refreshing on any 401 would spawn a browser login that cannot fix anything.

**A malformed token does not trigger renewal.** The sidecar overwrites the file
wholesale, so a login *could* repair local corruption. We still decline,
because a garbled file would otherwise turn every request into a refresh
attempt. The failure stays loud instead of becoming a storm.

**Single-flight ownership belongs to the sidecar, never the caller.** The
caller spawns a sidecar and returns; the sidecar acquires a non-blocking
`flock` for its entire run and a losing sidecar exits as a no-op. An earlier
design had the proxy hold the lock across `subprocess.Popen`, which made the
child's own acquire fail — a 401 could then produce no refresh at all. The lock
file is never unlinked: a fresh inode lets a later caller acquire immediately
and defeats mutual exclusion.

**The proxy returns 503 and never retries the request itself.** The client
retries. The proxy has no way to know whether the browser login will need a Duo
push, so holding the request open is worse than asking the client to come back.

**Token replacement is atomic.** The sidecar writes a mode-0600 temp file in
the destination directory, fsyncs, then `os.replace()`s it. A plain write
followed by `chmod` let a concurrent reader see partial JSON, fall back to
`USER_TOKEN`, and silently use an expired token.

## Consequences

- Any caller can ask for a renewal without knowing about locks or subprocesses.
- Both frontends behave identically on a token fault.
- The sidecar remains the only component that knows how to drive a browser.
- Concurrent 401s may spawn several short-lived sidecars. That is cheap and
  self-correcting; a lock held across process boundaries was not.
