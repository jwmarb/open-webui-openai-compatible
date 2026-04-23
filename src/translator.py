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


def create_openai_error(
    message: str,
    error_type: str = "invalid_request_error",
    code: int | None = None,
) -> dict:
    """Create an OpenAI-compatible error response body."""
    return {"error": {"message": message, "type": error_type, "code": code}}
