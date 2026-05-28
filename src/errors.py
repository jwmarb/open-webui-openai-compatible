"""Shared error handling for all proxy frontends (OpenAI, Anthropic)."""

from __future__ import annotations

import logging
from typing import Any

import openai

from .proxy.openai.models import OpenAIErrorDetail, OpenAIErrorResponse

logger = logging.getLogger(__name__)

__all__ = [
    "classify_upstream_error",
    "create_openai_error",
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


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: int | None = None,
) -> dict[str, Any]:
    """Build an OpenAI-format error dict."""
    resp = OpenAIErrorResponse(error=OpenAIErrorDetail(message=message, type=error_type, code=code))
    return resp.model_dump()
