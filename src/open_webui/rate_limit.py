"""Upstream rate-limit detection and stall scheduling (shared by both frontends).

The gateway throttles each end user to a fixed number of requests per window
and reports overage as HTTP 400 with a ``Rate limit exceeded`` detail string
— not a 429. The string carries the exact window-reset time. Verified against
genai.arizona.edu 2026-09-18: 20 requests per 60 s per end user. See
ADR-0006.

This module is transport-neutral like ``errors.py``: it owns detection and
sleep scheduling. Each frontend owns the wire format of the eventual 429.
"""

from __future__ import annotations

import math
import re
import time
from datetime import UTC, datetime
from typing import Final

import openai

__all__ = ["RateLimitStall", "is_rate_limit", "seconds_until_reset"]

# Wake this long before the advertised reset so small clock skew can never make
# us miss the window opening; one extra rejected request is free — a rejection
# consumes no slot and re-advertises the reset.
_WAKE_MARGIN_SECONDS: Final[float] = 0.5

# Fallback sleep when the gateway does not advertise a reset timestamp.
_FALLBACK_SECONDS: Final[float] = 1.0

_RESET_RE: Final[re.Pattern[str]] = re.compile(
    r"Limit resets at: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)


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
        """Seconds to sleep before retrying, or ``None`` to surface the error."""
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
        if reset_in is None:
            sleep = _FALLBACK_SECONDS
        else:
            sleep = max(0.0, reset_in - _WAKE_MARGIN_SECONDS)
        return min(sleep, remaining)

    def retry_after(self) -> int:
        """``Retry-After`` for the exhaustion response.

        Whole seconds until the last advertised reset, at least 1; 1 when the
        gateway advertised none.
        """
        reset_in = self._last_reset
        if reset_in is None:
            return 1
        return max(1, math.ceil(reset_in))
