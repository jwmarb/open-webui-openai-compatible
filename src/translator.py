from __future__ import annotations

THINKING_SUFFIX_EXTENDED = ":extended"
THINKING_SUFFIX_ADAPTIVE = ":adaptive"

EXTENDED_THINKING_CONFIG = {"type": "enabled", "budget_tokens": 32000}
EXTENDED_THINKING_CONFIG_SMALL = {"type": "enabled", "budget_tokens": 16000}
ADAPTIVE_THINKING_CONFIG = {"type": "adaptive"}

MIN_MAX_TOKENS_EXTENDED = 64000
MIN_MAX_TOKENS_EXTENDED_SMALL = 32000


def _is_claude_model(model_id: str) -> bool:
    return "claude" in model_id.lower()


def _is_small_context_claude(model_id: str) -> bool:
    """Haiku models have smaller max output (64k) so need a smaller budget."""
    return "haiku" in model_id.lower()


def _supports_adaptive(model_id: str) -> bool:
    """Adaptive thinking is supported on Opus 4.6+, Sonnet 4.6+, but NOT Haiku 4.5."""
    return _is_claude_model(model_id) and not _is_small_context_claude(model_id)


def generate_thinking_variants(model: dict) -> list[dict]:
    model_id = model.get("id", "")
    if not _is_claude_model(model_id):
        return []

    variants = []

    extended_id = model_id + THINKING_SUFFIX_EXTENDED
    variants.append({
        "id": extended_id,
        "object": "model",
        "created": model.get("created", 0),
        "owned_by": model.get("owned_by", ""),
    })

    if _supports_adaptive(model_id):
        adaptive_id = model_id + THINKING_SUFFIX_ADAPTIVE
        variants.append({
            "id": adaptive_id,
            "object": "model",
            "created": model.get("created", 0),
            "owned_by": model.get("owned_by", ""),
        })

    return variants


def resolve_thinking_model(model: str) -> tuple[str, dict | None]:
    if model.endswith(THINKING_SUFFIX_EXTENDED):
        base = model.removesuffix(THINKING_SUFFIX_EXTENDED)
        if _is_small_context_claude(base):
            return base, EXTENDED_THINKING_CONFIG_SMALL
        return base, EXTENDED_THINKING_CONFIG

    if model.endswith(THINKING_SUFFIX_ADAPTIVE):
        base = model.removesuffix(THINKING_SUFFIX_ADAPTIVE)
        return base, ADAPTIVE_THINKING_CONFIG

    return model, None


def apply_thinking_params(body: dict, thinking_config: dict) -> dict:
    body = {**body, "thinking": thinking_config}

    budget = thinking_config.get("budget_tokens", 0)
    if budget > 0:
        min_tokens = MIN_MAX_TOKENS_EXTENDED_SMALL if budget <= 16000 else MIN_MAX_TOKENS_EXTENDED
    else:
        min_tokens = MIN_MAX_TOKENS_EXTENDED

    current_max = body.get("max_tokens") or body.get("max_completion_tokens") or 0
    if current_max < min_tokens:
        body["max_tokens"] = min_tokens

    return body


def translate_models_response(raw: dict) -> dict:
    """Translate GET /api/models response to OpenAI /v1/models format.

    Input: {"data": [{"id": "...", "name": "...", "owned_by": "...", ...extra fields}]}
    Output: {"object": "list", "data": [{"id": "...", "object": "model", "created": 0, "owned_by": "..."}]}
    """
    items = raw.get("data", [])
    data = []
    for item in items:
        model = {
            "id": item.get("id", ""),
            "object": "model",
            "created": int(item.get("created", 0) or 0),
            "owned_by": item.get("owned_by", ""),
        }
        data.append(model)
        data.extend(generate_thinking_variants(model))
    return {"object": "list", "data": data}


# https://platform.openai.com/docs/api-reference/chat/create
OPENAI_CHAT_PARAMS: set[str] = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "user",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "response_format",
    "seed",
    "service_tier",
    "metadata",
    "store",
    "reasoning_effort",
    "functions",
    "function_call",
}


def sanitize_chat_body(body: dict) -> dict:
    return {k: v for k, v in body.items() if k in OPENAI_CHAT_PARAMS}


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: int | None = None,
) -> dict:
    """Create an OpenAI-compatible error response body."""
    return {"error": {"message": message, "type": error_type, "code": code}}
