"""FastAPI application composition.

`create_app()` is the seam: it accepts the upstream clients so tests can supply
fakes without patching module namespaces. `app` is the default production
instance for uvicorn.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx
import openai
from fastapi import FastAPI

from .auth import get_current_token
from .proxy.anthropic.routes import router as anthropic_router
from .proxy.openai.routes import router as openai_router
from .settings import Settings, settings

_pkg_logger = logging.getLogger("src")
_pkg_logger.setLevel(getattr(logging, settings.log_level))
if not _pkg_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s — %(message)s"))
    _pkg_logger.addHandler(_handler)

__all__ = ["UpstreamClients", "app", "build_upstream_clients", "create_app"]


class _TokenAuth(httpx.Auth):
    """Re-reads the token per request so a refresh takes effect without a restart."""

    def auth_flow(self, request: httpx.Request):
        request.headers["Authorization"] = f"Bearer {get_current_token()}"
        yield request


@dataclass(slots=True)
class UpstreamClients:
    models: httpx.AsyncClient
    chat: openai.AsyncOpenAI

    async def aclose(self) -> None:
        await self.chat.close()
        await self.models.aclose()


def build_upstream_clients(config: Settings) -> UpstreamClients:
    auth = _TokenAuth()
    timeout = httpx.Timeout(float(config.request_timeout), connect=10.0)

    models_client = httpx.AsyncClient(base_url=config.open_webui_url, timeout=timeout, auth=auth)
    chat_http = httpx.AsyncClient(base_url=f"{config.open_webui_url}/api", timeout=timeout, auth=auth)

    return UpstreamClients(
        models=models_client,
        chat=openai.AsyncOpenAI(
            api_key="proxy-auth-via-hook",
            base_url=f"{config.open_webui_url}/api",
            http_client=chat_http,
            max_retries=0,
        ),
    )


def create_app(
    config: Settings | None = None,
    clients: UpstreamClients | None = None,
) -> FastAPI:
    config = config or settings

    @asynccontextmanager
    async def _lifespan(instance: FastAPI) -> AsyncIterator[None]:
        upstream = clients or build_upstream_clients(config)
        instance.state.models_client = upstream.models
        instance.state.openai_client = upstream.chat
        try:
            yield
        finally:
            await upstream.aclose()

    instance = FastAPI(title="OpenAI-Compatible Proxy", lifespan=_lifespan)
    instance.include_router(openai_router)
    instance.include_router(anthropic_router)

    @instance.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return instance


app = create_app()
