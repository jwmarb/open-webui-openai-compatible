"""Anthropic response and error shapes used at runtime.

Request-side and SSE-event shapes are intentionally absent: `translate_request`
and `StreamingState` operate on raw dicts, so declaring inert models here would
imply validation that does not happen.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class AnthropicUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class AnthropicResponse(BaseModel):
    """Non-streaming Anthropic Messages API response."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[dict[str, Any]]
    model: str
    stop_reason: str | None = None
    stop_sequence: str | None = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class AnthropicErrorDetail(BaseModel):
    type: str = "invalid_request_error"
    message: str


class AnthropicErrorResponse(BaseModel):
    """Anthropic-format error response."""

    type: Literal["error"] = "error"
    error: AnthropicErrorDetail
