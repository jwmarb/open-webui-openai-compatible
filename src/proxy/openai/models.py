"""Pydantic response types matching the OpenAI API schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class OpenAIModel(BaseModel):
    """A single model entry in the OpenAI /v1/models response."""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = ""


class OpenAIModelList(BaseModel):
    """Response shape for GET /v1/models."""

    object: str = "list"
    data: list[OpenAIModel]


class ThinkingConfig(BaseModel):
    """Claude thinking configuration injected into chat completion requests."""

    type: Literal["enabled", "adaptive"]
    budget_tokens: int | None = None


class OpenAIErrorDetail(BaseModel):
    """Error detail nested inside an OpenAI error response."""

    message: str
    type: str = "invalid_request_error"
    code: int | None = None


class OpenAIErrorResponse(BaseModel):
    """Top-level OpenAI-compatible error response."""

    error: OpenAIErrorDetail
