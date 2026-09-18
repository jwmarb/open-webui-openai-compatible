"""Upstream rate-limit detection and stall scheduling (shared by both frontends).

Verified against genai.arizona.edu (Open WebUI 0.9.6 fronting a LiteLLM
gateway) on 2026-09-18. The gateway enforces two tiers, both reported as
HTTP 400 with a ``Rate limit exceeded`` detail string (a raw 429 is also
accepted):

- an **end-user tier**: 20 requests per 60 s, shared across all chat
  endpoints and models, keyed on the JWT identity. Its detail carries the
  exact window-reset time ("...Limit resets at: YYYY-MM-DD HH:MM:SS UTC")
  and the stall sleeps until it.
- a **burst tier**: "Rate limit exceeded: 10 requests per minute."
  10 per 60 s, shared across all chat endpoints and models (verified
  2026-09-18: a v1 burst blocks core; a core burst blocks v1). No reset
  timestamp is advertised, and the window boundary is not observable from
  the detail string. This is the binding constraint (10/min < 20/min), so
  the effective single-user ceiling is ~10 req/min. The proxy's job is to
  make that ceiling transparent: queue, stall, and deliver a smooth stream
  with no client-visible 400s.

``RateLimitStall`` therefore schedules two ways: to the advertised reset,
or to the next free slot computed by ``SlidingWindowTracker`` from this
process's own admission history. The tracker models the burst window as
rolling — each admission expires one window after it was made — which is
deliberately conservative: the measured behaviour (a full burst of 10 is
blocked for ~15-60 s, never released early) is consistent with a window
anchored at the first admission, and rolling can only sleep longer than
that, never shorter. Rejected requests consume no slot, which makes the
conservative sleep safe — a too-early retry is a free 400 that
re-advertises the state. See ADR-0006 and ADR-0007.


This module is transport-neutral like ``errors.py``: it owns detection and
sleep scheduling. Each frontend owns the wire format of the eventual 429.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from datetime import UTC, datetime
from typing import Final

import openai

__all__ = [
    "RateLimitStall",
    "SlidingWindowTracker",
    "is_rate_limit",
    "record_upstream_admission",
    "record_upstream_rejection",
    "rpm_tier_from_detail",
    "seconds_until_reset",
]

# Wake this long before the advertised reset so small clock skew can never make
# us miss the window opening; one extra rejected request is free — a rejection
# consumes no slot and re-advertises the reset.
_WAKE_MARGIN_SECONDS: Final[float] = 0.5

# Fallback sleep when the gateway does not advertise a reset timestamp.
_FALLBACK_SECONDS: Final[float] = 1.0

_RESET_RE: Final[re.Pattern[str]] = re.compile(
    r"Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)

# The gateway's sliding request tier advertises no reset timestamp — only
# the limit and unit ("Rate limit exceeded: 10 requests per minute."). The
# defaults are the observed genai.arizona.edu values (verified 2026-09-18);
# ``SlidingWindowTracker.observe()`` adopts whatever the detail string says.
_DEFAULT_RPM_LIMIT: Final[int] = 10
_DEFAULT_RPM_WINDOW: Final[float] = 60.0

# How early to fire ahead of the computed next slot. An early fire costs one
# rejected request, and a rejection consumes no slot, so the margin stays
# small.
_SLOT_MARGIN_SECONDS: Final[float] = 0.25

_RPM_TIER_RE: Final[re.Pattern[str]] = re.compile(
    r"Rate limit exceeded: (\d+) requests per (second|minute|hour)"
)
_RPM_TIER_UNITS: Final[dict[str, float]] = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
}


def _detail_of(exc: Exception) -> str:
    """Return the upstream ``detail`` string from the exception body, else ``""``."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
    return ""


def is_rate_limit(exc: Exception) -> bool:
    """Whether an upstream error is a rate limit.

    A 429 is a rate limit on its face. A 400 is one only when the gateway's
    "Rate limit exceeded" detail is present — a bare 400 is a client bug and
    must never stall.
    """
    if not isinstance(exc, openai.APIStatusError):
        return False
    if exc.status_code == 429:
        return True
    if exc.status_code == 400:
        return "rate limit exceeded" in _detail_of(exc).lower()
    return False


def seconds_until_reset(exc: Exception) -> float | None:
    """Seconds until the advertised window reset, or ``None`` if not advertised.

    Negative means the reset is already in the past (clock skew); treat it as
    "retry now".
    """
    match = _RESET_RE.search(_detail_of(exc))
    if match is None:
        return None
    try:
        reset_at = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=UTC
        )
    except ValueError:
        return None
    return (reset_at - datetime.now(UTC)).total_seconds()


def rpm_tier_from_detail(detail: str) -> tuple[int, float] | None:
    """Parse the sliding-tier detail into ``(limit, window_seconds)``.

    Matches "Rate limit exceeded: 10 requests per minute." — the tier that
    advertises no reset timestamp. Returns ``None`` for the end-user tier
    ("Rate limit exceeded for end_user: ...") and for anything else, so a
    bare 400 never re-shapes the window model.
    """
    match = _RPM_TIER_RE.search(detail)
    if match is None:
        return None
    return int(match.group(1)), _RPM_TIER_UNITS[match.group(2).lower()]


class SlidingWindowTracker:
    """Model the gateway's 10-per-minute burst window from this process's admissions.

    The gateway admits at most ``limit`` chat requests per ``window``
    seconds for the end user. The window boundary is not advertised; what
    is measured (verified 2026-09-18: a full burst of 10 is blocked for
    ~15-60 s, never released early) is that a request fired after ``limit``
    admissions within the window is rejected, and a rejected request
    consumes no slot. The tracker therefore models the window as rolling —
    each admission expires ``window`` seconds after it was made — so the
    computed next slot (oldest admission + window) is a conservative bound:
    whatever the gateway's true window anchor is, the tracker can only
    over-sleep, never fire early. An early fire costs one rejected request,
    and a rejection is free, so the conservatism is safe.

    The proxy is the sole consumer of the token, so its own admission
    history is a valid model of the gateway's window. An admission is
    recorded when an upstream request is accepted (the routes call
    ``record_upstream_admission()`` at each pre-header seam, including the
    first-chunk pre-read) and removed again when the gateway rejects it for
    rate limiting.
    """

    def __init__(self) -> None:
        self._admissions: deque[float] = deque()
        self._limit: int = _DEFAULT_RPM_LIMIT
        self._window: float = _DEFAULT_RPM_WINDOW

    def reset(self) -> None:
        """Clear history and adopt the defaults (test seam)."""
        self._admissions.clear()
        self._limit = _DEFAULT_RPM_LIMIT
        self._window = _DEFAULT_RPM_WINDOW

    def observe(self, limit: int, window: float) -> None:
        """Adopt the limit and window advertised by the latest detail."""
        self._limit = max(1, int(limit))
        self._window = max(0.0, float(window))

    @property
    def has_history(self) -> bool:
        """Whether any admission has been recorded since the last reset."""
        return bool(self._admissions)

    def mark_admitted(self) -> float:
        """Record an accepted upstream request; return its monotonic stamp."""
        now = time.monotonic()
        self._admissions.append(now)
        return now

    def mark_rejected(self, admitted_at: float) -> None:
        """Drop the admission record of a request the gateway rejected."""
        try:
            self._admissions.remove(admitted_at)
        except ValueError:
            pass

    def seconds_until_next_slot(self) -> float:
        """Seconds until the next slot frees; 0.0 when a slot is free now."""
        now = time.monotonic()
        while self._admissions and now - self._admissions[0] >= self._window:
            self._admissions.popleft()
        if len(self._admissions) < self._limit:
            return 0.0
        return max(0.0, self._admissions[0] + self._window - now)


_tracker: Final[SlidingWindowTracker] = SlidingWindowTracker()


def record_upstream_admission() -> float:
    """Record that an upstream chat request was accepted; return its stamp."""
    return _tracker.mark_admitted()


def record_upstream_rejection(admitted_at: float) -> None:
    """Record that the request admitted at *admitted_at* was rate-limit rejected."""
    _tracker.mark_rejected(admitted_at)

class RateLimitStall:
    """Per-request stall budget against upstream rate limits.

    One instance per incoming request. ``sleep_for()`` returns the seconds to
    sleep before the next upstream attempt, or ``None`` if the error is not a
    rate limit or the budget is spent. The budget starts on first detection:
    a request that never hits the limit is unstalled (the usual per-attempt
    ``REQUEST_TIMEOUT`` applies as before).
    """

    def __init__(self, budget_seconds: float) -> None:
        self._budget = max(0.0, budget_seconds)
        self._deadline: float | None = None
        self._last_detail: str = ""
        self._last_reset: float | None = None

    @property
    def last_detail(self) -> str:
        """The detail text of the last rate-limit response, ``""`` if none."""
        return self._last_detail

    @property
    def exhausted(self) -> bool:
        """True if a rate limit was seen and the stall budget is spent."""
        return self._deadline is not None and time.monotonic() >= self._deadline

    def sleep_for(self, exc: Exception) -> float | None:
        """Seconds to sleep before retrying, or ``None`` to surface the error.

        The sleep targets the advertised reset when the detail carries one;
        for the sliding tier (no reset timestamp) it targets the next free
        slot computed from this process's own admission history.
        """
        if self._budget <= 0 or not is_rate_limit(exc):
            return None
        now = time.monotonic()
        if self._deadline is None:
            self._deadline = now + self._budget
        detail = _detail_of(exc)
        if detail:
            self._last_detail = detail
        self._last_reset = seconds_until_reset(exc)
        remaining = self._deadline - now
        if remaining <= 0:
            return None
        reset_in = self._last_reset
        if reset_in is not None:
            sleep = max(0.0, reset_in - _WAKE_MARGIN_SECONDS)
        else:
            sleep = self._sliding_tier_sleep()
        return min(sleep, remaining)

    def _sliding_tier_sleep(self) -> float:
        """Sleep for the sliding request tier, which advertises no reset time.

        The detail carries the limit and unit, so the tracker adopts them;
        the next free slot is then computed from this process's admission
        history. With no history (fresh process, cold start) there is nothing
        to model, so the 1 s fallback stands in — the first accepted requests
        build the history that makes later stalls exact.
        """
        tier = rpm_tier_from_detail(self._last_detail)
        if tier is not None:
            _tracker.observe(*tier)
        if not _tracker.has_history:
            return _FALLBACK_SECONDS
        return max(_tracker.seconds_until_next_slot() - _SLOT_MARGIN_SECONDS, 0.05)

    def retry_after(self) -> int:
        """``Retry-After`` for the exhaustion response.

        Whole seconds until the last advertised reset, at least 1; 1 when the
        gateway advertised none.
        """
        reset_in = self._last_reset
        if reset_in is not None:
            return max(1, math.ceil(reset_in))
        if _tracker.has_history:
            return max(1, math.ceil(_tracker.seconds_until_next_slot()))
        return 1
