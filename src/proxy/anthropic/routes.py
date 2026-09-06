"""Anthropic Messages API route handler (POST /v1/messages)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Final

import openai
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...errors import classify_upstream_error, log_upstream_error
from ...settings import settings
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


_RETRY_BACKOFF_CAP: Final[int] = 120


async def _handle_streaming(
    ai_client: openai.AsyncOpenAI,
    sdk_kwargs: dict[str, Any],
    extra: dict[str, Any],
    model: str,
) -> JSONResponse | StreamingResponse:
    max_retries = settings.stream_empty_retry_max
    attempt = 0

    while True:
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
            if isinstance(exc, openai.APIStatusError) and 400 <= exc.status_code < 500:
                return _anthropic_error_response(exc, "Anthropic streaming first chunk")
            attempt += 1
            if attempt <= max_retries:
                sleep = min(1 << attempt, _RETRY_BACKOFF_CAP)
                logger.warning(
                    "Anthropic first-chunk error (attempt %d/%d): %s — retrying in %d seconds...",
                    attempt, max_retries, exc, sleep,
                )
                await asyncio.sleep(sleep)
                continue
            if isinstance(exc, StopAsyncIteration):
                logger.warning("Anthropic empty stream retries exhausted")
                state = StreamingState(model=model)
                final_events = state.finalize()

                async def _empty_stream() -> AsyncGenerator[bytes, None]:
                    for event in final_events:
                        yield _sse_event(event["type"], event)

                return StreamingResponse(
                    _empty_stream(),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            return _anthropic_error_response(exc, "Anthropic streaming first chunk")
        break

    async def _generate() -> AsyncGenerator[bytes, None]:
        state = StreamingState(model=model)
        chunk_dict = first_chunk.model_dump(exclude_unset=True)
        for event in state.translate_chunk(chunk_dict):
            yield _sse_event(event["type"], event)

        try:
            async for chunk in stream:  # type: ignore[union-attr]
                chunk_dict = chunk.model_dump(exclude_unset=True)
                for event in state.translate_chunk(chunk_dict):
                    yield _sse_event(event["type"], event)
        except Exception as exc:
            log_upstream_error(exc, "Anthropic mid-stream error")
            msg, _, _ = classify_upstream_error(exc)
            err_event = {"type": "error", "error": {"type": "api_error", "message": msg}}
            yield _sse_event("error", err_event)
            return

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
