"""OpenAI-compatible route handlers for /v1/models and /v1/chat/completions."""

from __future__ import annotations

import asyncio
import fcntl
import json
import logging
import os
import subprocess
import sys
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Final

import httpx
import openai
import openai.types.chat
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ...auth import is_token_expired_or_invalid
from ...errors import classify_upstream_error, create_openai_error, log_upstream_error
from ...settings import settings
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
REFRESH_LOCK_PATH: Final[Path] = Path("/tmp/openwebui-proxy-refresh.lock")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REFRESH_SCRIPT: Final[str] = str(PROJECT_ROOT / "playwright_login.py")

router = APIRouter()


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


_TOKEN_REJECTION_CODES = frozenset({"invalid_issuer", "invalid_token", "token_expired"})


def _extract_error_code(body: object) -> str:
    if not isinstance(body, dict):
        return ""
    code = body.get("code")
    if isinstance(code, str):
        return code
    inner = body.get("error")
    if isinstance(inner, dict):
        inner_code = inner.get("code")
        if isinstance(inner_code, str):
            return inner_code
    return ""


def _should_refresh_token(token: str | None, body: object) -> bool:
    """A 401 alone is ambiguous — refresh only on positive evidence the token is at fault.

    Upstream returns 401 for non-token reasons too (revoked model access, rate policy),
    and refreshing on those would spawn a browser login that cannot fix anything.
    """
    if is_token_expired_or_invalid(token):
        return True
    return _extract_error_code(body) in _TOKEN_REJECTION_CODES


def _trigger_refresh() -> None:
    """Spawn the refresh sidecar asynchronously, guarded by a lock file."""
    try:
        lock_fd = open(REFRESH_LOCK_PATH, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.info("Token refresh already in progress (lock held), skipping")
        return
    except OSError as exc:
        logger.error("Failed to acquire refresh lock: %s", exc)
        return

    def _release_lock(fd):
        fcntl.flock(fd, fcntl.LOCK_UN)
        fd.close()
        REFRESH_LOCK_PATH.unlink(missing_ok=True)

    try:
        script_path = _REFRESH_SCRIPT
        if not os.path.exists(script_path):
            script_path = "playwright-login.py"
        log_path = Path("/tmp/sidecar.log")
        subprocess.Popen(
            [sys.executable, script_path],
            stdout=open(log_path, "a"),
            stderr=open(log_path, "a"),
            start_new_session=True,
        )
        logger.info("Token refresh sidecar spawned")
    except Exception as exc:
        logger.error("Failed to spawn refresh sidecar: %s", exc)
        _release_lock(lock_fd)
    else:
        _release_lock(lock_fd)


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
        raw = await request.app.state.web_client.get_models()
        translated = translate_models_response(raw)
        logger.debug("GET /v1/models — returning %d models",
                      len(translated.get("data", [])))
        return JSONResponse(content=translated)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 401:
            from ...auth import get_current_token

            token = get_current_token()
            try:
                err_body = exc.response.json()
            except Exception:
                err_body = None
            if _should_refresh_token(token, err_body):
                _trigger_refresh()
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
        except openai.APIStatusError as exc:
            if exc.status_code == 401:
                from ...auth import get_current_token

                token = get_current_token()
                if _should_refresh_token(token, getattr(exc, "body", None)):
                    _trigger_refresh()
                    return _token_expired_error_response()
            return _upstream_error_response(exc, "Streaming create")
        except Exception as exc:
            return _upstream_error_response(exc, "Streaming create")

        # Pre-read the first chunk to detect empty streams and immediate errors.
        first_chunk: openai.types.chat.ChatCompletionChunk | None = None
        try:
            first_chunk = await stream.__anext__()  # type: ignore[union-attr]
        except openai.APIStatusError as exc:
            if exc.status_code == 401:
                from ...auth import get_current_token

                token = get_current_token()
                if _should_refresh_token(token, getattr(exc, "body", None)):
                    _trigger_refresh()
                    return _token_expired_error_response()
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
    except openai.APIStatusError as exc:
        if exc.status_code == 401:
            from ...auth import get_current_token

            token = get_current_token()
            if _should_refresh_token(token, getattr(exc, "body", None)):
                _trigger_refresh()
                return _token_expired_error_response()
        return _upstream_error_response(exc, "Non-streaming")
    except Exception as exc:
        return _upstream_error_response(exc, "Non-streaming")
