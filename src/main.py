"""FastAPI proxy exposing OpenAI-compatible endpoints backed by Open WebUI."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import httpx
import openai
from fastapi import FastAPI

from .client import WebClient
from .proxy.openai.routes import router as openai_router
from .settings import settings

_pkg_logger = logging.getLogger("src")
_pkg_logger.setLevel(getattr(logging, settings.log_level))
if not _pkg_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
    _pkg_logger.addHandler(_handler)

__all__ = ["app"]


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.web_client = WebClient(
        settings.open_webui_url,
        settings.user_token,
        request_timeout=settings.request_timeout,
    )
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
app.include_router(openai_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
