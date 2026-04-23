"""FastAPI proxy exposing OpenAI-compatible endpoints backed by Open WebUI."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Final

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .client import WebClient
from .settings import settings
from .translator import (
    apply_thinking_params,
    create_openai_error,
    resolve_thinking_model,
    sanitize_chat_body,
    translate_models_response,
)

logger = logging.getLogger(__name__)

_SSE_HEADERS: Final[dict[str, str]] = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}

_FINISH_REASON_NEEDLE = b'"finish_reason"'
_FINISH_REASON_NULL = b'"finish_reason":null'
_FINISH_REASON_NONE = b'"finish_reason": null'


def _extract_finish_reason(chunk: bytes) -> str | None:
    """Extract finish_reason from an SSE data chunk, if present and non-null.

    Scans raw bytes to avoid JSON parsing on every chunk.  Only parses JSON
    for the rare chunk that actually contains a non-null finish_reason.
    """
    if _FINISH_REASON_NEEDLE not in chunk:
        return None
    if _FINISH_REASON_NULL in chunk or _FINISH_REASON_NONE in chunk:
        return None
    for line in chunk.split(b"\n"):
        if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
            continue
        try:
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            for choice in choices:
                reason = choice.get("finish_reason")
                if reason is not None:
                    return reason
        except (json.JSONDecodeError, AttributeError):
            continue
    return None


__all__ = ["app"]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.web_client = WebClient(settings.open_webui_url, settings.user_token)
    yield
    await app.state.web_client.aclose()


app = FastAPI(title="OpenAI-Compatible Proxy", lifespan=_lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/models")
async def models(request: Request) -> JSONResponse:
    try:
        raw = await request.app.state.web_client.get_models()
        translated = translate_models_response(raw)
        return JSONResponse(content=translated)
    except httpx.HTTPStatusError as exc:
        err = create_openai_error("Upstream request failed", "api_error", exc.response.status_code)
        return JSONResponse(content=err, status_code=exc.response.status_code)
    except Exception:
        err = create_openai_error("Upstream service unavailable", "server_error", 502)
        return JSONResponse(content=err, status_code=502)


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    body: dict[str, Any] = sanitize_chat_body(await request.json())
    logger.debug("Incoming request body keys: %s", list(body.keys()))

    model = body.get("model", "")
    base_model, thinking_config = resolve_thinking_model(model)
    if thinking_config is not None:
        body["model"] = base_model
        body = apply_thinking_params(body, thinking_config)

    is_stream = body.get("stream") is True

    if is_stream:
        stream_gen = await request.app.state.web_client.post_chat_completion(
            body, stream=True
        )

        # Pre-read the first chunk so errors raised before any data
        # (e.g. upstream 400 on raise_for_status) surface here where
        # we can still return a proper HTTP error response.
        first_chunk: bytes | None = None
        try:
            first_chunk = await anext(stream_gen.__aiter__(), None)
        except httpx.HTTPStatusError as exc:
            err = create_openai_error(
                "Upstream request failed", "api_error", exc.response.status_code,
            )
            return JSONResponse(content=err, status_code=exc.response.status_code)
        except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            logger.error("Upstream timeout during stream pre-read: %s", exc)
            err = create_openai_error(
                "Upstream request timed out", "timeout_error", 504,
            )
            return JSONResponse(content=err, status_code=504)
        except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
            logger.error("Upstream connection error during stream pre-read: %s", exc)
            err = create_openai_error(
                "Upstream connection failed", "api_error", 502,
            )
            return JSONResponse(content=err, status_code=502)
        except Exception as exc:
            logger.exception("Unexpected error during stream pre-read: %s", type(exc).__name__)
            err = create_openai_error(
                "Upstream service unavailable", "server_error", 502,
            )
            return JSONResponse(content=err, status_code=502)

        async def event_stream():
            finish_reason: str | None = None
            try:
                if first_chunk is not None:
                    finish_reason = _extract_finish_reason(first_chunk) or finish_reason
                    yield first_chunk
                async for chunk in stream_gen:
                    finish_reason = _extract_finish_reason(chunk) or finish_reason
                    yield chunk
                logger.info(
                    "Stream ended for model=%s finish_reason=%s",
                    model, finish_reason,
                )
            except httpx.HTTPStatusError as exc:
                err = create_openai_error(
                    "Upstream request failed", "api_error", exc.response.status_code,
                )
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except (httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                logger.error("Upstream timeout mid-stream: %s", exc)
                err = create_openai_error(
                    "Upstream request timed out", "timeout_error", 504,
                )
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except (httpx.RemoteProtocolError, httpx.ReadError) as exc:
                logger.error("Upstream connection error mid-stream: %s", exc)
                err = create_openai_error(
                    "Upstream connection failed", "api_error", 502,
                )
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("Unexpected error mid-stream: %s", type(exc).__name__)
                err = create_openai_error(
                    "Upstream service unavailable", "server_error", 502,
                )
                yield f"data: {json.dumps(err)}\n\n".encode()
                yield b"data: [DONE]\n\n"
            except BaseException as exc:
                # CancelledError (client disconnect) and other BaseExceptions.
                # Log so silent stream deaths become visible.
                logger.warning(
                    "Stream interrupted by %s for model=%s",
                    type(exc).__name__, model,
                )
                raise
            finally:
                await stream_gen.aclose()

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers=_SSE_HEADERS,
        )

    try:
        result = await request.app.state.web_client.post_chat_completion(body)
        return JSONResponse(content=result)
    except httpx.HTTPStatusError as exc:
        err = create_openai_error("Upstream request failed", "api_error", exc.response.status_code)
        return JSONResponse(content=err, status_code=exc.response.status_code)
    except Exception:
        err = create_openai_error("Upstream service unavailable", "server_error", 502)
        return JSONResponse(content=err, status_code=502)
