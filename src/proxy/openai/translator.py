"""Translation layer between Open WebUI and OpenAI API formats.

Handles model list translation, request body rewriting (Bedrock tool-field
scrubbing, stream usage injection), and Claude thinking variant logic.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from .models import OpenAIModel, OpenAIModelList, ThinkingConfig

logger = logging.getLogger(__name__)

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
    "generate_thinking_variants",
    "resolve_thinking_model",
    "rewrite_chat_body",
    "sanitize_chat_body",
    "translate_models_response",
]


def _normalize_model_id(model_id: str) -> str:
    """Lowercase and unify separators so ``anthropic.claude-x`` and ``bedrock_claude_x`` agree."""
    base, _ = _split_thinking_suffix(model_id)
    return re.sub(r"[._/]", "-", base.lower())


# Open WebUI's GET /api/models returns the raw upstream entry (id/object/created/
# owned_by) with no capability metadata, so ID matching is the only available
# signal. Allowlist, not "not OpenAI": an unrecognised model must never receive a
# provider-specific parameter. Add new Anthropic families here as they ship.
_ANTHROPIC_FAMILY_TOKENS: frozenset[str] = frozenset({
    "claude",
    "fable",
    "mythos",
})

# Anthropic families that accept thinking.type="adaptive". Claude 4.5 and earlier
# accept only type="enabled" and reject adaptive with 400 "adaptive thinking is
# not supported on this model", so this is an allowlist of known-capable families
# rather than the previous "everything except Haiku".
_ADAPTIVE_MIN_VERSION: tuple[int, int] = (4, 6)
# Opus/Sonnet 4.7+ (and Fable/Mythos) go further than merely *accepting* adaptive:
# they REJECT thinking.type="enabled" with 400 '"thinking.type.enabled" is not
# supported for this model. Use "thinking.type.adaptive" and
# "output_config.effort"'. Verified against genai.arizona.edu: claude-4-6-opus and
# claude-4-6-sonnet accept "enabled" (200), claude-5-opus refuses it (400).
_ADAPTIVE_ONLY_MIN_VERSION: tuple[int, int] = (4, 7)
_ADAPTIVE_CAPABLE_LINES: frozenset[str] = frozenset({"opus", "sonnet"})
_ADAPTIVE_ALWAYS_CAPABLE: frozenset[str] = frozenset({"fable", "mythos"})

# Matches a major[-minor] version anywhere in a normalized ID: "claude-4-6-opus",
# "claude-sonnet-4-6", "claude-opus-5". Minor defaults to 0 when absent. Bounded to
# two digits so trailing date stamps ("...-4-6-20250514") are not read as versions.
_VERSION_PATTERN = re.compile(r"(?<!\d)(\d{1,2})(?:-(\d{1,2}))?(?!\d)")


def _extract_version(normalized_id: str) -> tuple[int, int] | None:
    matches = _VERSION_PATTERN.findall(normalized_id)
    if not matches:
        return None
    major, minor = matches[0]
    return int(major), int(minor or 0)


def _is_claude_model(model_id: str) -> bool:
    """True when the model belongs to an Anthropic family that supports ``thinking``."""
    normalized = _normalize_model_id(model_id)
    return any(token in normalized for token in _ANTHROPIC_FAMILY_TOKENS)


def _is_small_context_claude(model_id: str) -> bool:
    """Haiku models have smaller max output (64k) so need a smaller budget."""
    return "haiku" in _normalize_model_id(model_id)


def _supports_adaptive(model_id: str) -> bool:
    """True only for Anthropic families documented to accept ``type="adaptive"``."""
    return _adaptive_capability(model_id) >= _ADAPTIVE_MIN_VERSION


def _requires_adaptive(model_id: str) -> bool:
    """True for families that reject ``type="enabled"`` and accept only adaptive."""
    return _adaptive_capability(model_id) >= _ADAPTIVE_ONLY_MIN_VERSION


def _adaptive_capability(model_id: str) -> tuple[int, int]:
    """Version used for adaptive gating; ``(0, 0)`` when thinking does not apply.

    Fable/Mythos carry no version digits but share the newest request surface, so
    they report a version above every numeric gate.
    """
    if not _is_claude_model(model_id):
        return (0, 0)

    normalized = _normalize_model_id(model_id)
    if any(token in normalized for token in _ADAPTIVE_ALWAYS_CAPABLE):
        return (99, 99)

    if not any(line in normalized for line in _ADAPTIVE_CAPABLE_LINES):
        return (0, 0)

    return _extract_version(normalized) or (0, 0)


def generate_thinking_variants(model: OpenAIModel) -> list[OpenAIModel]:
    if not _is_claude_model(model.id):
        return []

    variants: list[OpenAIModel] = []

    variants.append(
        OpenAIModel(
            id=model.id + THINKING_SUFFIX_EXTENDED,
            created=model.created,
            owned_by=model.owned_by,
        )
    )

    if _supports_adaptive(model.id):
        variants.append(
            OpenAIModel(
                id=model.id + THINKING_SUFFIX_ADAPTIVE,
                created=model.created,
                owned_by=model.owned_by,
            )
        )

    return variants


def _split_thinking_suffix(model: str) -> tuple[str, str | None]:
    for suffix in (THINKING_SUFFIX_EXTENDED, THINKING_SUFFIX_ADAPTIVE):
        if model.endswith(suffix):
            return model.removesuffix(suffix), suffix
    return model, None


def resolve_thinking_model(model: str) -> tuple[str, ThinkingConfig | None]:
    """Strip a thinking suffix and return the config it maps to, if the model supports it."""
    base, suffix = _split_thinking_suffix(model)
    if suffix is None:
        return model, None

    if not _is_claude_model(base):
        logger.warning(
            "Ignoring %s suffix on non-Anthropic model %r: thinking is Anthropic-only",
            suffix, base,
        )
        return base, None

    if suffix == THINKING_SUFFIX_ADAPTIVE:
        if not _supports_adaptive(base):
            logger.warning(
                "Ignoring %s suffix on %r: model does not support adaptive thinking",
                suffix, base,
            )
            return base, None
        return base, ADAPTIVE_THINKING_CONFIG

    if _requires_adaptive(base):
        logger.info(
            "Mapping %s suffix on %r to adaptive thinking: this family rejects "
            "thinking.type='enabled' and requires adaptive + output_config.effort",
            suffix, base,
        )
        return base, ADAPTIVE_THINKING_CONFIG

    if _is_small_context_claude(base):
        return base, EXTENDED_THINKING_CONFIG_SMALL
    return base, EXTENDED_THINKING_CONFIG


def apply_thinking_params(body: dict[str, Any], thinking_config: ThinkingConfig) -> dict[str, Any]:
    body = {**body, "thinking": thinking_config.model_dump(exclude_none=True)}

    budget = thinking_config.budget_tokens or 0
    if budget > 0:
        min_tokens = max(budget * 2, MIN_MAX_TOKENS_EXTENDED_SMALL)
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
            body["tools"] = [_DUMMY_TOOL.copy()]
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
            body["stream_options"] = {**opts, "include_usage": True}
        else:
            body["stream_options"] = {"include_usage": True}
    return body


def _strip_incompatible_thinking(body: dict[str, Any]) -> dict[str, Any]:
    """Reconcile a client-supplied ``thinking`` param with what the model accepts.

    ``thinking`` is Anthropic-only. Open WebUI forwards unknown top-level params
    verbatim on non-Azure OpenAI connections, so sending it to an OpenAI-family
    model reaches the provider and hard-fails with
    400 ``unknown_parameter: 'thinking'``, so it is dropped. Adaptive-only Claude
    families reject ``type="enabled"`` just as hard, so that is coerced to
    adaptive. Both keep the request usable instead of turning a recoverable call
    into a retry loop.
    """
    if "thinking" not in body:
        return body

    model = body.get("model", "")
    if not _is_claude_model(model):
        body.pop("thinking", None)
        logger.warning(
            "Stripped client-supplied 'thinking' param for non-Anthropic model %r "
            "(parameter is Anthropic-only and would be rejected upstream)",
            model,
        )
        return body

    thinking = body.get("thinking")
    if (
        _requires_adaptive(model)
        and isinstance(thinking, dict)
        and thinking.get("type") == "enabled"
    ):
        body["thinking"] = ADAPTIVE_THINKING_CONFIG.model_dump(exclude_none=True)
        logger.warning(
            "Coerced client-supplied thinking.type='enabled' to 'adaptive' for %r "
            "(family rejects enabled thinking; use output_config.effort for depth)",
            model,
        )

    return body


_UNSUPPORTED_FIELDS: frozenset[str] = frozenset(
    {
        "vector_store_ids",
        "file_ids",
    }
)

def _strip_unsupported_fields(body: dict[str, Any]) -> dict[str, Any]:
    """Remove fields that upstream providers (e.g. Bedrock) reject as extra inputs."""
    for field in _UNSUPPORTED_FIELDS:
        body.pop(field, None)
    return body


def _inject_chat_id(body: dict[str, Any]) -> dict[str, Any]:
    """Inject a ``local:``-prefixed ephemeral chat_id.

    Open WebUI 0.9.x requires a non-None ``chat_id`` string on
    ``/api/chat/completions`` (``NoneType.startswith`` crash). The
    ``local:`` prefix tells OWUI to skip all DB persistence — no
    conversation rows, no ownership checks, no history lookup.
    """
    body["chat_id"] = f"local:{uuid.uuid4()}"
    return body


def _strip_session_id(body: dict[str, Any]) -> dict[str, Any]:
    """Strip ``session_id`` to prevent OWUI multi-model fan-out.

    When ``session_id`` is present, OWUI routes through its WebSocket
    task pool (multi-model fan-out). The proxy has no WebSocket
    connection — fan-out would hang or fail.
    """
    body.pop("session_id", None)
    return body


def rewrite_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    rewritten = {**body}
    rewritten = _strip_unsupported_fields(rewritten)
    rewritten = _strip_incompatible_thinking(rewritten)
    rewritten = _scrub_bedrock_tool_fields(rewritten)
    rewritten = _ensure_stream_usage(rewritten)
    rewritten = _inject_chat_id(rewritten)
    rewritten = _strip_session_id(rewritten)
    return rewritten


sanitize_chat_body = rewrite_chat_body
