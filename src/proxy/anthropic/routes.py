"""Anthropic Messages API route handler (POST /v1/messages)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Final

import openai
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...errors import classify_upstream_error, log_upstream_error
from ..openai.routes import _split_body_for_sdk
from ..openai.translator import rewrite_chat_body
from .translator import (
    StreamingState,
    create_anthropic_error,
    translate_request,
    translate_response,
)

logger = logging.getLogger(__name__)

_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

router = APIRouter()


def _anthropic_error_response(exc: Exception, context: str) -> JSONResponse:
    log_upstream_error(exc, context)
    msg, etype, code = classify_upstream_error(exc)
    anthropic_type_map = {
        "api_error": "api_error",
        "timeout_error": "api_error",
        "server_error": "api_error",
        "invalid_request_error": "invalid_request_error",
    }
    return JSONResponse(
        content=create_anthropic_error(msg, anthropic_type_map.get(etype, "api_error")),
        status_code=code,
    )


def _sse_event(event_type: str, data: dict[str, Any]) -> bytes:
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


@router.post("/v1/messages", response_model=None)
async def messages(request: Request) -> JSONResponse | StreamingResponse:
    raw_body = await request.json()
    logger.info(
        "POST /v1/messages — model=%s stream=%s messages=%d",
        raw_body.get("model", "?"),
        raw_body.get("stream", False),
        len(raw_body.get("messages", [])),
    )

    requested_model = raw_body.get("model", "")
    is_stream = raw_body.get("stream", False)

    openai_body = translate_request(raw_body)
    openai_body = rewrite_chat_body(openai_body)

    ai_client: openai.AsyncOpenAI = request.app.state.openai_client
    sdk_kwargs, extra = _split_body_for_sdk(openai_body)

    if is_stream:
        return await _handle_streaming(ai_client, sdk_kwargs, extra, requested_model)
    return await _handle_non_streaming(ai_client, sdk_kwargs, extra, requested_model)


async def _handle_streaming(
    ai_client: openai.AsyncOpenAI,
    sdk_kwargs: dict[str, Any],
    extra: dict[str, Any],
    model: str,
) -> JSONResponse | StreamingResponse:
    try:
        stream = await ai_client.chat.completions.create(
            **sdk_kwargs,
            extra_body=extra or None,
        )
    except Exception as exc:
        return _anthropic_error_response(exc, "Anthropic streaming create")

    first_chunk: Any = None
    try:
        first_chunk = await stream.__anext__()  # type: ignore[union-attr]
    except Exception as exc:
        return _anthropic_error_response(exc, "Anthropic streaming first chunk")

    async def _generate() -> AsyncGenerator[bytes, None]:
        state = StreamingState(model=model)
        chunk_dict = first_chunk.model_dump(exclude_unset=True)
        events = state.translate_chunk(chunk_dict)
        for event in events:
            yield _sse_event(event["type"], event)

        saw_finish = False
        try:
            async for chunk in stream:  # type: ignore[union-attr]
                chunk_dict = chunk.model_dump(exclude_unset=True)
                events = state.translate_chunk(chunk_dict)
                for event in events:
                    yield _sse_event(event["type"], event)
                    if event.get("type") == "message_stop":
                        saw_finish = True
        except Exception as exc:
            log_upstream_error(exc, "Anthropic mid-stream error")
            msg, _, _ = classify_upstream_error(exc)
            err_event = {"type": "error", "error": {"type": "api_error", "message": msg}}
            yield _sse_event("error", err_event)
            return

        if not saw_finish:
            for event in state.finalize():
                yield _sse_event(event["type"], event)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _handle_non_streaming(
    ai_client: openai.AsyncOpenAI,
    sdk_kwargs: dict[str, Any],
    extra: dict[str, Any],
    model: str,
) -> JSONResponse:
    try:
        result = await ai_client.chat.completions.create(
            **sdk_kwargs,
            extra_body=extra or None,
        )
        response_dict = result.model_dump(exclude_unset=True)
        anthropic_response = translate_response(response_dict, model)
        return JSONResponse(content=anthropic_response)
    except Exception as exc:
        return _anthropic_error_response(exc, "Anthropic non-streaming")
