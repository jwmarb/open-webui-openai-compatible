"""FastAPI proxy exposing OpenAI-compatible endpoints backed by Open WebUI."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx
import openai
import openai.types.chat
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import WebClient
from .settings import settings
from .translator import (
    apply_thinking_params,
    create_openai_error,
    resolve_thinking_model,
    rewrite_chat_body,
    translate_models_response,
)

# Configure the "src" namespace logger so all src.* modules emit output.
# Uses the "src" parent rather than root to avoid flooding with third-party noise.
_pkg_logger = logging.getLogger("src")
_pkg_logger.setLevel(getattr(logging, settings.log_level))
if not _pkg_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
    _pkg_logger.addHandler(_handler)

logger = logging.getLogger(__name__)

_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

# Fields the openai SDK's chat.completions.create() accepts as explicit keyword
# args.  Everything else in the rewritten body goes into ``extra_body``.
_SDK_KNOWN_PARAMS: Final[frozenset[str]] = frozenset({
    "model", "messages", "stream",
    "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "max_tokens", "max_completion_tokens", "n", "presence_penalty",
    "response_format", "seed", "stop", "temperature", "top_p",
    "tools", "tool_choice", "parallel_tool_calls", "user",
    "stream_options", "metadata", "store", "service_tier",
})

_RETRY_BACKOFF_CAP: Final[int] = 120


def _split_body_for_sdk(body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split into (sdk_kwargs, extra_body) based on ``_SDK_KNOWN_PARAMS``."""
    sdk_kwargs: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    for key, value in body.items():
        if key in _SDK_KNOWN_PARAMS:
            sdk_kwargs[key] = value
        else:
            extra[key] = value
    return sdk_kwargs, extra


# ---------------------------------------------------------------------------
# Upstream error classification
# ---------------------------------------------------------------------------

def _classify_upstream_error(exc: Exception) -> tuple[str, str, int]:
    """Map an upstream exception to (message, error_type, http_status)."""
    if isinstance(exc, openai.APIStatusError):
        return "Upstream request failed", "api_error", exc.status_code
    if isinstance(exc, openai.APITimeoutError):
        return "Upstream request timed out", "timeout_error", 504
    if isinstance(exc, (openai.APIConnectionError, openai.APIError)):
        return "Upstream connection failed", "api_error", 502
    return "Upstream service unavailable", "server_error", 502


def _log_upstream_error(exc: Exception, context: str) -> None:
    """Log with traceback for unexpected errors, single line for known ones."""
    if isinstance(exc, (openai.APIStatusError, openai.APITimeoutError,
                        openai.APIConnectionError, openai.APIError)):
        logger.error("%s: %s", context, exc)
    else:
        logger.exception("%s", context)


def _upstream_error_response(exc: Exception, context: str) -> JSONResponse:
    """Build a JSONResponse from an upstream exception."""
    _log_upstream_error(exc, context)
    msg, etype, code = _classify_upstream_error(exc)
    return JSONResponse(content=create_openai_error(msg, etype, code), status_code=code)


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
        _log_upstream_error(exc, "Mid-stream error")
        msg, etype, code = _classify_upstream_error(exc)
        yield _make_error_sse(msg, etype, code)
        yield _SSE_DONE
        return

    if seen_finish_reason is None:
        yield _make_finish_reason_sse(last_model, last_chunk_id)
    yield _SSE_DONE


__all__ = ["app"]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # WebClient is retained for /v1/models only.
    app.state.web_client = WebClient(
        settings.open_webui_url,
        settings.user_token,
        request_timeout=settings.request_timeout,
    )
    # AsyncOpenAI client for /v1/chat/completions.
    # base_url points at /api so the SDK's /chat/completions path maps to
    # Open WebUI's /api/chat/completions endpoint.
    app.state.openai_client = openai.AsyncOpenAI(
        api_key=settings.user_token,
        base_url=f"{settings.open_webui_url}/api",
        timeout=httpx.Timeout(float(settings.request_timeout), connect=10.0),
        max_retries=0,
    )
    yield
    await app.state.openai_client.close()
    await app.state.web_client.aclose()


app = FastAPI(title="OpenAI-Compatible Proxy", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    try:
        logger.debug("GET /v1/models — fetching upstream model list")
        raw = await request.app.state.web_client.get_models()
        translated = translate_models_response(raw)
        logger.debug("GET /v1/models — returning %d models",
                     len(translated.get("data", [])))
        return JSONResponse(content=translated)
    except httpx.HTTPStatusError as exc:
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


# ---------------------------------------------------------------------------
# Chat completions endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", response_model=None)
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
    if thinking_config is not None:
        logger.info("Thinking variant detected: %s → base=%s config=%s",
                    model, base_model, thinking_config)
        body["model"] = base_model
        body = apply_thinking_params(body, thinking_config)

    is_stream = body.get("stream") is True
    logger.debug("Forwarding to upstream: model=%s stream=%s",
                 body.get("model"), is_stream)

    ai_client: openai.AsyncOpenAI = request.app.state.openai_client
    sdk_kwargs, extra = _split_body_for_sdk(body)

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
    attempt = 0

    while True:
        try:
            stream = await ai_client.chat.completions.create(
                **sdk_kwargs,
                extra_body=extra or None,
            )
        except Exception as exc:
            return _upstream_error_response(exc, "Streaming create")

        # Pre-read the first chunk to detect empty streams and immediate errors.
        first_chunk: openai.types.chat.ChatCompletionChunk | None = None
        try:
            first_chunk = await stream.__anext__()  # type: ignore[union-attr]
        except Exception as exc:
            # Client errors (4xx) are non-retryable — the request itself is wrong.
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
    try:
        result = await ai_client.chat.completions.create(
            **sdk_kwargs,
            extra_body=extra or None,
        )
        response_dict = result.model_dump(exclude_unset=True)
        logger.info(
            "Non-streaming response received: model=%s choices=%d",
            response_dict.get("model", "?"),
            len(response_dict.get("choices", [])),
        )
        return JSONResponse(content=response_dict)
    except Exception as exc:
        return _upstream_error_response(exc, "Non-streaming")
