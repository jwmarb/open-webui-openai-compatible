"""Open WebUI request policy.

Everything this gateway requires of a chat body regardless of which wire
format the client spoke. Both frontends hand a canonical OpenAI-shaped body to
`prepare_chat_body()`; none of these rules belong to the OpenAI protocol.

Pass order is load-bearing — see `rewrite_chat_body`.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Final

from .capabilities import ModelCapabilities, capabilities_for

logger = logging.getLogger(__name__)

__all__ = [
    "SDK_KNOWN_PARAMS",
    "prepare_chat_body",
    "rewrite_chat_body",
    "split_body_for_sdk",
]

# Fields the openai SDK's chat.completions.create() accepts as explicit keyword
# args. Everything else in the rewritten body goes into ``extra_body``.
SDK_KNOWN_PARAMS: Final[frozenset[str]] = frozenset({
    "model", "messages", "stream",
    "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "max_tokens", "max_completion_tokens", "n", "presence_penalty",
    "response_format", "seed", "stop", "temperature", "top_p",
    "tools", "tool_choice", "parallel_tool_calls", "user",
    "stream_options", "metadata", "store", "service_tier",
})

# The gateway rejects a max_tokens too small to hold a thinking block, so any
# request that carries (or is given) adaptive thinking needs this headroom.
# Shared by the :adaptive variant path (proxy.openai.translator) and the
# reasoning-effort injection in _reconcile_thinking.
MIN_MAX_TOKENS_ADAPTIVE: Final[int] = 64_000

_UNSUPPORTED_FIELDS: Final[frozenset[str]] = frozenset({"vector_store_ids", "file_ids"})

_REASONING_CONTROL_FIELDS: Final[tuple[str, ...]] = (
    "reasoning_effort",
    "reasoning",
    "effort",
    "verbosity",
    "textVerbosity",
    "thinking",
    "output_config",
)

_DUMMY_TOOL: Final[dict[str, Any]] = {
    "type": "function",
    "function": {
        "name": "dummy_tool",
        "description": "placeholder tool — never call",
        "parameters": {"type": "object", "properties": {}},
    },
}


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


def _scrub_bedrock_tool_fields(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
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


def _ensure_stream_usage(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    if body.get("stream") is True:
        opts = body.get("stream_options")
        if isinstance(opts, dict):
            body["stream_options"] = {**opts, "include_usage": True}
        else:
            body["stream_options"] = {"include_usage": True}
    return body


def _meaningful_effort(value: Any) -> str | None:
    """Return the depth value if it is a non-empty, non-'none' effort string."""
    if isinstance(value, str) and value.strip().lower() not in ("", "none"):
        return value
    return None


def _reconcile_thinking(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    """Reconcile the client's thinking/reasoning controls with what the model accepts.

    Two jobs, both load-bearing:

    1. *Legality.* ``thinking`` is Anthropic-only, so it is dropped for
       OpenAI-family models (upstream 400s on the unknown top-level param).
       Adaptive-only families reject ``type="enabled"`` with the same 400, so it
       is coerced to ``type="adaptive"`` — a client ``display`` is kept, but
       ``budget_tokens`` is not, since the adaptive form takes no budget.

    2. *Visibility.* Claude 4.7+/5.x (and fable/mythos) default
       ``thinking.display`` to ``"omitted"``: the model reasons, but the text is
       withheld, so ``reasoning_tokens`` reads 0 and clients see no thinking.
       For those families we surface explicitly requested reasoning by adding
       ``display="summarized"`` — to an adaptive/enabled ``thinking``, or by
       synthesising one when a depth control (``reasoning_effort`` or
       ``output_config.effort``) was supplied without a ``thinking``. A
       client's own ``display`` (including ``"omitted"``) and a
       ``type="disabled"`` are left untouched.
    """
    if not caps.supports_thinking:
        if "thinking" in body:
            body.pop("thinking", None)
            logger.warning(
                "Stripped client-supplied 'thinking' param for non-Anthropic model %r "
                "(parameter is Anthropic-only and would be rejected upstream)",
                body.get("model", ""),
            )
        return body

    thinking = body.get("thinking")
    if isinstance(thinking, dict):
        if caps.requires_adaptive and thinking.get("type") == "enabled":
            coerced: dict[str, Any] = {"type": "adaptive"}
            if "display" in thinking:
                coerced["display"] = thinking["display"]
            body["thinking"] = coerced
            logger.warning(
                "Coerced client-supplied thinking.type='enabled' to 'adaptive' for %r "
                "(family rejects enabled thinking; use output_config.effort for depth)",
                body.get("model", ""),
            )
            thinking = coerced

        if (
            caps.defaults_to_omitted_thinking
            # `"enabled"` is defensive: requires_adaptive families coerce
            # enabled→adaptive above, so only adaptive reaches this today.
            and thinking.get("type") in ("adaptive", "enabled")
            and "display" not in thinking
        ):
            body["thinking"] = {**thinking, "display": "summarized"}
            logger.info(
                "Added thinking.display='summarized' for %r: this family hides its "
                "thinking text by default, so the reasoning would otherwise be lost",
                body.get("model", ""),
            )
        return body

    effort = _meaningful_effort(body.get("reasoning_effort"))
    if effort is None:
        output_config = body.get("output_config")
        if isinstance(output_config, dict):
            effort = _meaningful_effort(output_config.get("effort"))
    if caps.defaults_to_omitted_thinking and effort is not None:
        # No usable `thinking`: surface the depth request by synthesising visible
        # adaptive thinking, and give it the headroom the gateway requires.
        body["thinking"] = {"type": "adaptive", "display": "summarized"}
        effective_max = body.get("max_completion_tokens") or body.get("max_tokens") or 0
        if effective_max < MIN_MAX_TOKENS_ADAPTIVE:
            if "max_completion_tokens" in body:
                body["max_completion_tokens"] = MIN_MAX_TOKENS_ADAPTIVE
            else:
                body["max_tokens"] = MIN_MAX_TOKENS_ADAPTIVE
        logger.info(
            "Injected thinking={type: adaptive, display: summarized} for %r to make "
            "the requested reasoning effort visible (this family hides thinking text "
            "by default)",
            body.get("model", ""),
        )
    return body


def _strip_incompatible_reasoning_effort(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    """Drop every reasoning/verbosity control for models whose upstream rejects them."""
    if caps.accepts_reasoning_controls:
        return body

    dropped = [field for field in _REASONING_CONTROL_FIELDS if field in body]
    if not dropped:
        return body

    for field in dropped:
        body.pop(field, None)
    logger.warning(
        "Stripped %s for %r: this family is served via Bedrock Converse, which "
        "rejects reasoning and verbosity controls (depth stays at the default)",
        ", ".join(repr(field) for field in dropped),
        body.get("model", ""),
    )
    return body


def _strip_incompatible_effort_config(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    """Drop ``output_config.effort`` for Claude families that reject the field.

    Claude 4.5 and earlier answer it with 400 "This model does not support the
    effort parameter." 4.6 accepts and ignores it, 5.x honours it, so only the
    old families need the strip. A sibling ``format`` key is preserved — it is
    translated to ``response_format`` separately and is not an effort control.
    """
    if caps.accepts_effort_config:
        return body

    config = body.get("output_config")
    if not isinstance(config, dict) or "effort" not in config:
        return body

    remaining = {key: value for key, value in config.items() if key != "effort"}
    if remaining:
        body["output_config"] = remaining
    else:
        body.pop("output_config", None)
    logger.warning(
        "Stripped output_config.effort for %r: this family rejects the effort "
        "parameter (depth stays at the default)",
        body.get("model", ""),
    )
    return body


def _strip_unsupported_fields(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    for field in _UNSUPPORTED_FIELDS:
        body.pop(field, None)
    return body


def _inject_chat_id(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    """Inject a ``local:``-prefixed ephemeral chat_id.

    Open WebUI 0.9.x requires a non-None ``chat_id`` string (``NoneType.startswith``
    crash). The ``local:`` prefix tells it to skip all DB persistence — no
    conversation rows, no ownership checks, no history lookup.
    """
    body["chat_id"] = f"local:{uuid.uuid4()}"
    return body


def _strip_session_id(body: dict[str, Any], caps: ModelCapabilities) -> dict[str, Any]:
    """Strip ``session_id`` to prevent Open WebUI multi-model fan-out.

    With ``session_id`` present it routes through its WebSocket task pool. The
    proxy holds no WebSocket connection, so fan-out would hang.
    """
    body.pop("session_id", None)
    return body


_PASSES: Final[tuple[Any, ...]] = (
    _strip_unsupported_fields,
    _reconcile_thinking,
    _strip_incompatible_reasoning_effort,
    _strip_incompatible_effort_config,
    _scrub_bedrock_tool_fields,
    _ensure_stream_usage,
    _inject_chat_id,
    _strip_session_id,
)


def rewrite_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    """Apply every gateway requirement to a canonical chat body.

    Order matters: thinking reconciliation runs before the reasoning-control
    strip so an adaptive coercion can still be removed for families that reject
    all controls, the effort-config strip runs after it so a family rejecting
    every control has already lost ``output_config`` wholesale, and tool
    scrubbing runs before stream-usage injection.
    """
    caps = capabilities_for(body.get("model", ""))
    rewritten = {**body}
    for apply_pass in _PASSES:
        rewritten = apply_pass(rewritten, caps)
    return rewritten


def split_body_for_sdk(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split into (sdk_kwargs, extra_body) based on ``SDK_KNOWN_PARAMS``."""
    sdk_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in body.items():
        if key in SDK_KNOWN_PARAMS:
            sdk_kwargs[key] = value
        else:
            extra[key] = value
    return sdk_kwargs, extra


def prepare_chat_body(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Rewrite a canonical chat body and split it for the SDK."""
    return split_body_for_sdk(rewrite_chat_body(body))
