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
