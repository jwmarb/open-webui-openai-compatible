"""Pydantic types for the Anthropic Messages API request/response shapes."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request content block types
# ---------------------------------------------------------------------------


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ImageSource(BaseModel):
    type: Literal["base64", "url"]
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class ImageBlock(BaseModel):
    type: Literal["image"] = "image"
    source: ImageSource


class DocumentSource(BaseModel):
    type: str
    media_type: str | None = None
    data: str | None = None
    url: str | None = None


class DocumentBlock(BaseModel):
    type: Literal["document"] = "document"
    source: DocumentSource
    title: str | None = None


class ToolUseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Any = {}


class ToolResultContent(BaseModel):
    type: Literal["text", "image"] = "text"
    text: str | None = None
    source: ImageSource | None = None


class ToolResultBlock(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: str | list[ToolResultContent] | None = None
    is_error: bool | None = None


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


class RedactedThinkingBlock(BaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: str


# Union of all content block types used in messages
ContentBlock = (
    TextBlock
    | ImageBlock
    | DocumentBlock
    | ToolUseBlock
    | ToolResultBlock
    | ThinkingBlock
    | RedactedThinkingBlock
)


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class AnthropicMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[dict[str, Any]]


class AnthropicToolInputSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Any] | None = None
    required: list[str] | None = None


class AnthropicTool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=dict)


class AnthropicToolChoice(BaseModel):
    type: Literal["auto", "any", "none", "tool"]
    name: str | None = None


class AnthropicThinkingConfig(BaseModel):
    type: str
    budget_tokens: int | None = None


class AnthropicOutputConfig(BaseModel):
    effort: str | None = None
    format: dict[str, Any] | None = None


class AnthropicMetadata(BaseModel):
    user_id: str | None = None


class AnthropicRequest(BaseModel):
    """Top-level Anthropic Messages API request body."""

    model: str
    messages: list[AnthropicMessage]
    max_tokens: int = 4096
    system: str | list[dict[str, Any]] | None = None
    thinking: AnthropicThinkingConfig | None = None
    tools: list[AnthropicTool] | None = None
    tool_choice: AnthropicToolChoice | None = None
    stop_sequences: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    stream: bool = False
    metadata: AnthropicMetadata | dict[str, Any] | None = None
    output_config: AnthropicOutputConfig | None = None


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------


class TextResponseBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingResponseBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: str


class ToolUseResponseBlock(BaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str
    name: str
    input: Any = {}


ResponseContentBlock = TextResponseBlock | ThinkingResponseBlock | ToolUseResponseBlock


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


# ---------------------------------------------------------------------------
# Streaming event types
# ---------------------------------------------------------------------------


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: dict[str, Any]


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: dict[str, Any]


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: dict[str, Any]


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: dict[str, Any]
    usage: dict[str, Any] | None = None


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"
