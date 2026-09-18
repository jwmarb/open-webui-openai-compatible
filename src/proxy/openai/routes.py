"""OpenAI-compatible route handlers for /v1/models and /v1/chat/completions."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from typing import Any, Final

import httpx
import openai
import openai.types.chat
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...auth import get_current_token, request_refresh, should_refresh
from ...errors import classify_upstream_error, log_upstream_error
from ...open_webui.rate_limit import RateLimitStall
from ...open_webui.request_policy import split_body_for_sdk
from ...settings import settings
from .errors import create_openai_error
from .translator import (
    apply_thinking_params,
    resolve_thinking_model,
    rewrite_chat_body,
    translate_models_response,
)

logger = logging.getLogger(__name__)

_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

_RETRY_BACKOFF_CAP: Final[int] = 120

router = APIRouter()


# ---------------------------------------------------------------------------
# Upstream error response helper
# ---------------------------------------------------------------------------

def _upstream_error_response(exc: Exception, context: str) -> JSONResponse:
    """Build a JSONResponse from an upstream exception."""
    log_upstream_error(exc, context)
    msg, etype, code = classify_upstream_error(exc)
    return JSONResponse(content=create_openai_error(msg, etype, code), status_code=code)


def _token_expired_error_response() -> JSONResponse:
    """Build a 503 response indicating token expiry and refresh in progress."""
    err = create_openai_error(
        "Upstream authentication token expired. Token refresh initiated. Please retry your request.",
        "server_error",
        503,
    )
    return JSONResponse(content=err, status_code=503)
def _rate_limit_exhausted_response(stall: RateLimitStall) -> JSONResponse:
    """Build a 429 with Retry-After once the stall budget is spent (ADR-0006)."""
    msg = "Upstream is rate limiting and the stall budget is exhausted."
    if stall.last_detail:
        msg += f" Last upstream response: {stall.last_detail}"
    err = create_openai_error(msg, "rate_limit_error", 429)
    return JSONResponse(
        content=err,
        status_code=429,
        headers={"Retry-After": str(stall.retry_after())},
    )

def _refresh_for(exc: openai.APIStatusError) -> bool:
    """Ask the token store to renew when a 401 shows positive token-fault evidence."""
    token = get_current_token()
    if not should_refresh(token, getattr(exc, "body", None)):
        return False
    request_refresh()
    return True



# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

def _chunk_to_sse(chunk: openai.types.chat.ChatCompletionChunk) -> bytes:
    """Serialize a ``ChatCompletionChunk`` to an SSE data line."""
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n".encode()


def _extract_finish_reason_from_chunk(
    chunk: openai.types.chat.ChatCompletionChunk,
) -> str | None:
    """Return the first non-null ``finish_reason`` from *chunk*, or ``None``."""
    for choice in chunk.choices or []:
        if choice.finish_reason is not None:
            return choice.finish_reason
    return None


def _make_finish_reason_sse(model: str, chunk_id: str) -> bytes:
    """Build a synthetic SSE chunk with ``finish_reason='stop'``."""
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "model": model,
    }
    return f"data: {json.dumps(payload)}\n\n".encode()


def _make_error_sse(message: str, error_type: str, code: int) -> bytes:
    """Build an SSE error event in OpenAI JSON format."""
    err = create_openai_error(message, error_type, code)
    return f"data: {json.dumps(err)}\n\n".encode()


_SSE_DONE: Final[bytes] = b"data: [DONE]\n\n"


async def _stream_with_first(
    first: openai.types.chat.ChatCompletionChunk,
    rest: openai.AsyncStream[openai.types.chat.ChatCompletionChunk],
) -> AsyncGenerator[bytes, None]:
    """Yield SSE bytes for *first* chunk followed by all remaining *rest* chunks.

    Tracks ``finish_reason`` across chunks and synthesizes one if the upstream
    stream ends without sending it.
    """
    seen_finish_reason: str | None = None
    last_chunk_id = first.id or ""
    last_model = first.model or ""

    reason = _extract_finish_reason_from_chunk(first)
    if reason is not None:
        seen_finish_reason = reason
    yield _chunk_to_sse(first)

    try:
        async for chunk in rest:
            last_chunk_id = chunk.id or last_chunk_id
            last_model = chunk.model or last_model
            reason = _extract_finish_reason_from_chunk(chunk)
            if reason is not None:
                seen_finish_reason = reason
            yield _chunk_to_sse(chunk)
    except Exception as exc:
        log_upstream_error(exc, "Mid-stream error")
        msg, etype, code = classify_upstream_error(exc)
        yield _make_error_sse(msg, etype, code)
        yield _SSE_DONE
        return

    if seen_finish_reason is None:
        yield _make_finish_reason_sse(last_model, last_chunk_id)
    yield _SSE_DONE


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    try:
        logger.debug("GET /v1/models — fetching upstream model list")
        response = await request.app.state.models_client.get("/api/models")
        response.raise_for_status()
        raw = response.json()
        translated = translate_models_response(raw)
        logger.debug("GET /v1/models — returning %d models",
                      len(translated.get("data", [])))
        return JSONResponse(content=translated)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            try:
                err_body = exc.response.json()
            except Exception:
                err_body = None
            if should_refresh(get_current_token(), err_body):
                request_refresh()
                return _token_expired_error_response()
        logger.error("GET /v1/models — upstream HTTP %s",
                      exc.response.status_code)
        err = create_openai_error(
            "Upstream request failed", "api_error", exc.response.status_code)
        return JSONResponse(content=err, status_code=exc.response.status_code)
    except Exception:
        logger.exception("GET /v1/models — unexpected error")
        err = create_openai_error(
            "Upstream service unavailable", "server_error", 502)
        return JSONResponse(content=err, status_code=502)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    raw_body = await request.json()
    logger.info(
        "POST /v1/chat/completions — model=%s stream=%s messages=%d",
        raw_body.get("model", "?"),
        raw_body.get("stream", False),
        len(raw_body.get("messages", [])),
    )
    logger.debug("Raw request body keys: %s", list(raw_body.keys()))
    body: dict[str, Any] = rewrite_chat_body(raw_body)
    logger.debug("Sanitized body keys: %s", list(body.keys()))

    model = body.get("model", "")
    base_model, thinking_config = resolve_thinking_model(model)
    if base_model != model:
        body["model"] = base_model
    if thinking_config is not None:
        logger.info("Thinking variant detected: %s → base=%s config=%s",
                    model, base_model, thinking_config)
        body = apply_thinking_params(body, thinking_config)

    is_stream = body.get("stream") is True
    logger.debug("Forwarding to upstream: model=%s stream=%s",
                 body.get("model"), is_stream)

    ai_client: openai.AsyncOpenAI = request.app.state.openai_client
    sdk_kwargs, extra = split_body_for_sdk(body)

    if is_stream:
        return await _handle_streaming(ai_client, sdk_kwargs, extra)

    return await _handle_non_streaming(ai_client, sdk_kwargs, extra)


async def _handle_streaming(
    ai_client: openai.AsyncOpenAI,
    sdk_kwargs: dict[str, Any],
    extra: dict[str, Any],
) -> JSONResponse | StreamingResponse:
    """Handle a streaming chat completion request with empty-stream retry."""
    max_retries = settings.stream_empty_retry_max
    stall = RateLimitStall(settings.rate_limit_stall_max_seconds)
    attempt = 0

    while True:
        try:
            stream = await ai_client.chat.completions.create(
                **sdk_kwargs,
                extra_body=extra or None,
            )
        except openai.APIStatusError as exc:
            if exc.status_code == 401 and _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Streaming create: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
            return _upstream_error_response(exc, "Streaming create")
        except Exception as exc:
            return _upstream_error_response(exc, "Streaming create")

        # Pre-read the first chunk to detect empty streams and immediate errors.
        first_chunk: openai.types.chat.ChatCompletionChunk | None = None
        try:
            first_chunk = await stream.__anext__()  # type: ignore[union-attr]
        except openai.APIStatusError as exc:
            if exc.status_code == 401 and _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Streaming first chunk: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
            if 400 <= exc.status_code < 500:
                return _upstream_error_response(exc, "Streaming first chunk")
            attempt += 1
            if attempt <= max_retries:
                sleep = min(1 << attempt, _RETRY_BACKOFF_CAP)
                logger.warning(
                    "First-chunk error (attempt %d/%d): %s — retrying in %d seconds...",
                    attempt, max_retries, exc, sleep,
                )
                await asyncio.sleep(sleep)
                continue
            return _upstream_error_response(exc, "Streaming first chunk")
        except Exception as exc:
            if isinstance(exc, openai.APIStatusError):
                sleep = stall.sleep_for(exc)
                if sleep is not None:
                    logger.warning(
                        "Streaming first chunk: upstream rate limited — stalling %.1f s (budget %d s)",
                        sleep, settings.rate_limit_stall_max_seconds,
                    )
                    await asyncio.sleep(sleep)
                    continue
                if stall.exhausted:
                    return _rate_limit_exhausted_response(stall)
            if isinstance(exc, openai.APIStatusError) and 400 <= exc.status_code < 500:
                return _upstream_error_response(exc, "Streaming first chunk")
            attempt += 1
            if attempt <= max_retries:
                sleep = min(1 << attempt, _RETRY_BACKOFF_CAP)
                logger.warning(
                    "First-chunk error (attempt %d/%d): %s — retrying in %d seconds...",
                    attempt, max_retries, exc, sleep,
                )
                await asyncio.sleep(sleep)
                continue
            # Retries exhausted.
            if isinstance(exc, StopAsyncIteration):
                logger.warning(
                    "Empty stream retries exhausted, injecting stop")

                async def _empty_stream_response() -> AsyncGenerator[bytes, None]:
                    yield _make_finish_reason_sse(
                        sdk_kwargs.get("model", ""), "")
                    yield _SSE_DONE

                return StreamingResponse(
                    _empty_stream_response(),
                    media_type="text/event-stream",
                    headers=_SSE_HEADERS,
                )
            return _upstream_error_response(exc, "Streaming first chunk")

        # Got a real first chunk — stream it along with the rest.
        break

    assert first_chunk is not None
    return StreamingResponse(
        _stream_with_first(first_chunk, stream),  # type: ignore[arg-type]
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )


async def _handle_non_streaming(
    ai_client: openai.AsyncOpenAI,
    sdk_kwargs: dict[str, Any],
    extra: dict[str, Any],
) -> JSONResponse:
    """Handle a non-streaming chat completion request."""
    stall = RateLimitStall(settings.rate_limit_stall_max_seconds)
    while True:
        try:
            result = await ai_client.chat.completions.create(
                **sdk_kwargs,
                extra_body=extra or None,
            )
        except openai.APIStatusError as exc:
            if exc.status_code == 401 and _refresh_for(exc):
                return _token_expired_error_response()
            sleep = stall.sleep_for(exc)
            if sleep is not None:
                logger.warning(
                    "Non-streaming: upstream rate limited — stalling %.1f s (budget %d s)",
                    sleep, settings.rate_limit_stall_max_seconds,
                )
                await asyncio.sleep(sleep)
                continue
            if stall.exhausted:
                return _rate_limit_exhausted_response(stall)
            return _upstream_error_response(exc, "Non-streaming")
        except Exception as exc:
            return _upstream_error_response(exc, "Non-streaming")
        response_dict = result.model_dump(exclude_unset=True)
        logger.info(
            "Non-streaming response received: model=%s choices=%d",
            response_dict.get("model", "?"),
            len(response_dict.get("choices", [])),
        )
        return JSONResponse(content=response_dict)
