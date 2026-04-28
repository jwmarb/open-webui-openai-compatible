"""Translation layer between Open WebUI and OpenAI API formats.

Handles model list translation, request body rewriting (Bedrock tool-field
scrubbing, stream usage injection), and Claude thinking variant logic.
"""

from __future__ import annotations

from typing import Any

from .models import OpenAIErrorDetail, OpenAIErrorResponse, OpenAIModel, OpenAIModelList, ThinkingConfig

THINKING_SUFFIX_EXTENDED = ":extended"
THINKING_SUFFIX_ADAPTIVE = ":adaptive"

EXTENDED_THINKING_CONFIG = ThinkingConfig(type="enabled", budget_tokens=32_000)
EXTENDED_THINKING_CONFIG_SMALL = ThinkingConfig(type="enabled", budget_tokens=16_000)
ADAPTIVE_THINKING_CONFIG = ThinkingConfig(type="adaptive")

MIN_MAX_TOKENS_EXTENDED = 64_000
MIN_MAX_TOKENS_EXTENDED_SMALL = 32_000

_DUMMY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "dummy_tool",
        "description": "placeholder tool — never call",
        "parameters": {"type": "object", "properties": {}},
    },
}

__all__ = [
    "apply_thinking_params",
    "create_openai_error",
    "generate_thinking_variants",
    "resolve_thinking_model",
    "rewrite_chat_body",
    "sanitize_chat_body",
    "translate_models_response",
]


def _is_claude_model(model_id: str) -> bool:
    return "claude" in model_id.lower()


def _is_small_context_claude(model_id: str) -> bool:
    """Haiku models have smaller max output (64k) so need a smaller budget."""
    return "haiku" in model_id.lower()


def _supports_adaptive(model_id: str) -> bool:
    """Adaptive thinking is supported on Opus 4.6+, Sonnet 4.6+, but NOT Haiku 4.5."""
    return _is_claude_model(model_id) and not _is_small_context_claude(model_id)


def generate_thinking_variants(model: OpenAIModel) -> list[OpenAIModel]:
    if not _is_claude_model(model.id):
        return []

    variants: list[OpenAIModel] = []

    variants.append(OpenAIModel(
        id=model.id + THINKING_SUFFIX_EXTENDED,
        created=model.created,
        owned_by=model.owned_by,
    ))

    if _supports_adaptive(model.id):
        variants.append(OpenAIModel(
            id=model.id + THINKING_SUFFIX_ADAPTIVE,
            created=model.created,
            owned_by=model.owned_by,
        ))

    return variants


def resolve_thinking_model(model: str) -> tuple[str, ThinkingConfig | None]:
    if model.endswith(THINKING_SUFFIX_EXTENDED):
        base = model.removesuffix(THINKING_SUFFIX_EXTENDED)
        if _is_small_context_claude(base):
            return base, EXTENDED_THINKING_CONFIG_SMALL
        return base, EXTENDED_THINKING_CONFIG

    if model.endswith(THINKING_SUFFIX_ADAPTIVE):
        base = model.removesuffix(THINKING_SUFFIX_ADAPTIVE)
        return base, ADAPTIVE_THINKING_CONFIG

    return model, None


def apply_thinking_params(body: dict[str, Any], thinking_config: ThinkingConfig) -> dict[str, Any]:
    body = {**body, "thinking": thinking_config.model_dump(exclude_none=True)}

    budget = thinking_config.budget_tokens or 0
    if budget > 0:
        min_tokens = MIN_MAX_TOKENS_EXTENDED_SMALL if budget <= 16_000 else MIN_MAX_TOKENS_EXTENDED
    else:
        min_tokens = MIN_MAX_TOKENS_EXTENDED

    current_max = body.get("max_tokens") or body.get("max_completion_tokens") or 0
    if current_max < min_tokens:
        body["max_tokens"] = min_tokens

    return body


def translate_models_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate GET /api/models response to OpenAI /v1/models format."""
    items = raw.get("data", [])
    data: list[OpenAIModel] = []
    for item in items:
        model = OpenAIModel(
            id=item.get("id", ""),
            created=int(item.get("created", 0) or 0),
            owned_by=item.get("owned_by", ""),
        )
        data.append(model)
        data.extend(generate_thinking_variants(model))
    result = OpenAIModelList(data=data)
    return result.model_dump()


def _messages_reference_tools(messages: Any) -> bool:
    if not isinstance(messages, list):
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            return True
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and len(tool_calls) > 0:
            return True
        if msg.get("tool_call_id"):
            return True
    return False


def _scrub_bedrock_tool_fields(body: dict[str, Any]) -> dict[str, Any]:
    tools = body.get("tools")
    has_tools = isinstance(tools, list) and len(tools) > 0

    if not has_tools:
        body.pop("tools", None)
        body.pop("tool_choice", None)
        body.pop("parallel_tool_calls", None)

        if _messages_reference_tools(body.get("messages")):
            body["tools"] = [_DUMMY_TOOL]
    else:
        choice = body.get("tool_choice")
        if isinstance(choice, dict):
            choice_type = choice.get("type")
        elif isinstance(choice, str):
            choice_type = choice
        else:
            choice_type = None

        if choice_type == "none":
            body.pop("tools", None)
            body.pop("tool_choice", None)
            body.pop("parallel_tool_calls", None)
        elif choice_type in ("any", "required"):
            body["tool_choice"] = "auto"

    body.pop("functions", None)
    body.pop("function_call", None)
    return body


def _ensure_stream_usage(body: dict[str, Any]) -> dict[str, Any]:
    if body.get("stream") is True:
        opts = body.get("stream_options")
        if isinstance(opts, dict):
            opts["include_usage"] = True
        else:
            body["stream_options"] = {"include_usage": True}
    return body


def rewrite_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    rewritten = {**body}
    rewritten = _scrub_bedrock_tool_fields(rewritten)
    rewritten = _ensure_stream_usage(rewritten)
    return rewritten


sanitize_chat_body = rewrite_chat_body


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: int | None = None,
) -> dict[str, Any]:
    resp = OpenAIErrorResponse(error=OpenAIErrorDetail(message=message, type=error_type, code=code))
    return resp.model_dump()
