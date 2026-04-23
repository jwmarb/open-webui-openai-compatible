from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from .client import WebClient
from .translator import (
    translate_models_response,
    create_openai_error,
    sanitize_chat_body,
    resolve_thinking_model,
    apply_thinking_params,
)

logger = logging.getLogger(__name__)

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "X-Accel-Buffering": "no",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.web_client = WebClient()
    yield
    await app.state.web_client.aclose()


app = FastAPI(title="OpenAI-Compatible Proxy", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/v1/models")
async def models(request: Request):
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


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = sanitize_chat_body(await request.json())
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
            if first_chunk is not None:
                yield first_chunk
            try:
                async for chunk in stream_gen:
                    yield chunk
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
