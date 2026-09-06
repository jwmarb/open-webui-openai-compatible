"""OpenAI-format error responses."""

from __future__ import annotations

from typing import Any

from .models import OpenAIErrorDetail, OpenAIErrorResponse

__all__ = ["create_openai_error"]


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: int | None = None,
) -> dict[str, Any]:
    resp = OpenAIErrorResponse(error=OpenAIErrorDetail(message=message, type=error_type, code=code))
    return resp.model_dump()
