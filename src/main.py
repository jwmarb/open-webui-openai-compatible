from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import httpx

from .client import WebClient
from .translator import translate_models_response, create_openai_error


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
    body = await request.json()

    is_stream = body.get("stream") is True

    if is_stream:
        stream_gen = await request.app.state.web_client.post_chat_completion(
            body, stream=True
        )

        async def event_generator():
            try:
                async for line in stream_gen:
                    yield line + "\n"
            except httpx.HTTPStatusError as exc:
                err = create_openai_error(
                    "Upstream request failed", "api_error", exc.response.status_code,
                )
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception:
                err = create_openai_error(
                    "Upstream service unavailable", "server_error", 502,
                )
                yield f"data: {json.dumps(err)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
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
