"""Transport-neutral upstream error classification.

Wire-format error bodies belong to each frontend: see
`src.proxy.openai.errors` and `src.proxy.anthropic.translator`.
"""

from __future__ import annotations

import logging

import openai

logger = logging.getLogger(__name__)

__all__ = [
    "classify_upstream_error",
    "log_upstream_error",
]


def classify_upstream_error(exc: Exception) -> tuple[str, str, int]:
    """Map an upstream exception to (message, error_type, http_status)."""
    if isinstance(exc, openai.APIStatusError):
        return "Upstream request failed", "api_error", exc.status_code
    if isinstance(exc, openai.APITimeoutError):
        return "Upstream request timed out", "timeout_error", 504
    if isinstance(exc, (openai.APIConnectionError, openai.APIError)):
        return "Upstream connection failed", "api_error", 502
    return "Upstream service unavailable", "server_error", 502


def log_upstream_error(exc: Exception, context: str) -> None:
    """Log with traceback for unexpected errors, single line for known ones."""
    if isinstance(exc, (openai.APIStatusError, openai.APITimeoutError,
                        openai.APIConnectionError, openai.APIError)):
        logger.error("%s: %s", context, exc)
    else:
        logger.exception("%s", context)
