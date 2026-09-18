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

from ...auth import get_current_token, request_refresh, should_refresh
from ...errors import classify_upstream_error, log_upstream_error
from ...open_webui.rate_limit import (
    RateLimitStall,
    is_rate_limit,
    record_upstream_admission,
    record_upstream_rejection,
)
from ...open_webui.request_policy import prepare_chat_body
from ...settings import settings
from ..openai.translator import apply_thinking_params, resolve_thinking_model
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


def _refresh_for(exc: Exception) -> bool:
    """Renew on a 401 that carries positive evidence the token is at fault."""
    if not isinstance(exc, openai.APIStatusError) or exc.status_code != 401:
        return False
    if not should_refresh(get_current_token(), getattr(exc, "body", None)):
        return False
    request_refresh()
    return True


def _token_expired_error_response() -> JSONResponse:
    return JSONResponse(
        content=create_anthropic_error(
            "Upstream authentication token expired. Token refresh initiated. Please retry your request.",
            "api_error",
        ),
        status_code=503,
    )
def _rate_limit_exhausted_response(stall: RateLimitStall) -> JSONResponse:
    """Build a 429 with Retry-After once the stall budget is spent (ADR-0006)."""
    msg = "Upstream is rate limiting and the stall budget is exhausted."
    if stall.last_detail:
        msg += f" Last upstream response: {stall.last_detail}"
    return JSONResponse(
        content=create_anthropic_error(msg, "rate_limit_error"),
        status_code=429,
        headers={"Retry-After": str(stall.retry_after())},
    )

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

    base_model, thinking_config = resolve_thinking_model(openai_body.get("model", ""))
    openai_body["model"] = base_model
    if thinking_config is not None:
        openai_body = apply_thinking_params(openai_body, thinking_config)

    ai_client: openai.AsyncOpenAI = request.app.state.openai_client
    sdk_kwargs, extra = prepare_chat_body(openai_body)

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
    stall = RateLimitStall(settings.rate_limit_stall_max_seconds)
    attempt = 0

    while True:
        admitted_at = record_upstream_admission()
        try:
            stream = await ai_client.chat.completions.create(
                **sdk_kwargs,
                extra_body=extra or None,
            )
        except Exception as exc:
            if is_rate_limit(exc):
                record_upstream_rejection(admitted_at)
            if _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Anthropic streaming create: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
            return _anthropic_error_response(exc, "Anthropic streaming create")

        first_chunk: Any = None
        try:
            first_chunk = await stream.__anext__()  # type: ignore[union-attr]
        except Exception as exc:
            if is_rate_limit(exc):
                record_upstream_rejection(admitted_at)
            if _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Anthropic streaming first chunk: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
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
    stall = RateLimitStall(settings.rate_limit_stall_max_seconds)
    while True:
        admitted_at = record_upstream_admission()
        try:
            result = await ai_client.chat.completions.create(
                **sdk_kwargs,
                extra_body=extra or None,
            )
        except Exception as exc:
            if is_rate_limit(exc):
                record_upstream_rejection(admitted_at)
            if _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Anthropic non-streaming: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
            return _anthropic_error_response(exc, "Anthropic non-streaming")
        response_dict = result.model_dump(exclude_unset=True)
        anthropic_response = translate_response(response_dict, model)
        return JSONResponse(content=anthropic_response)
