"""OpenAI wire-format translation.

Model-list translation and the Claude thinking-variant surface. Gateway request
policy lives in `src.open_webui.request_policy`; model capability rules live in
`src.open_webui.capabilities`.
"""

from __future__ import annotations

import logging
from typing import Any

from ...open_webui.capabilities import (
    THINKING_SUFFIX_ADAPTIVE,
    THINKING_SUFFIX_EXTENDED,
    capabilities_for,
    split_thinking_suffix,
)
from ...open_webui.request_policy import rewrite_chat_body
from .models import OpenAIModel, OpenAIModelList, ThinkingConfig

logger = logging.getLogger(__name__)

EXTENDED_THINKING_CONFIG = ThinkingConfig(type="enabled", budget_tokens=32_000)
EXTENDED_THINKING_CONFIG_SMALL = ThinkingConfig(type="enabled", budget_tokens=16_000)
ADAPTIVE_THINKING_CONFIG = ThinkingConfig(type="adaptive")

MIN_MAX_TOKENS_EXTENDED = 64_000
MIN_MAX_TOKENS_EXTENDED_SMALL = 32_000

__all__ = [
    "ADAPTIVE_THINKING_CONFIG",
    "EXTENDED_THINKING_CONFIG",
    "EXTENDED_THINKING_CONFIG_SMALL",
    "MIN_MAX_TOKENS_EXTENDED",
    "MIN_MAX_TOKENS_EXTENDED_SMALL",
    "THINKING_SUFFIX_ADAPTIVE",
    "THINKING_SUFFIX_EXTENDED",
    "apply_thinking_params",
    "generate_thinking_variants",
    "resolve_thinking_model",
    "rewrite_chat_body",
    "sanitize_chat_body",
    "translate_models_response",
]


def generate_thinking_variants(model: OpenAIModel) -> list[OpenAIModel]:
    if not capabilities_for(model.id).is_anthropic:
        return []

    variants: list[OpenAIModel] = []

    variants.append(
        OpenAIModel(
            id=model.id + THINKING_SUFFIX_EXTENDED,
            created=model.created,
            owned_by=model.owned_by,
        )
    )

    if capabilities_for(model.id).supports_adaptive:
        variants.append(
            OpenAIModel(
                id=model.id + THINKING_SUFFIX_ADAPTIVE,
                created=model.created,
                owned_by=model.owned_by,
            )
        )

    return variants


def resolve_thinking_model(model: str) -> tuple[str, ThinkingConfig | None]:
    """Strip a thinking suffix and return the config it maps to, if the model supports it."""
    base, suffix = split_thinking_suffix(model)
    if suffix is None:
        return model, None

    caps = capabilities_for(base)
    if not caps.is_anthropic:
        logger.warning(
            "Ignoring %s suffix on non-Anthropic model %r: thinking is Anthropic-only",
            suffix, base,
        )
        return base, None

    if suffix == THINKING_SUFFIX_ADAPTIVE:
        if not caps.supports_adaptive:
            logger.warning(
                "Ignoring %s suffix on %r: model does not support adaptive thinking",
                suffix, base,
            )
            return base, None
        return base, ADAPTIVE_THINKING_CONFIG

    if caps.requires_adaptive:
        logger.info(
            "Mapping %s suffix on %r to adaptive thinking: this family rejects "
            "thinking.type='enabled' and requires adaptive + output_config.effort",
            suffix, base,
        )
        return base, ADAPTIVE_THINKING_CONFIG

    if caps.small_context:
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


sanitize_chat_body = rewrite_chat_body
