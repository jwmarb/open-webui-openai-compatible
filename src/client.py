"""Async HTTP client for Open WebUI upstream API."""

from __future__ import annotations

import logging
from typing import Any

import httpx

_MAX_ERROR_LOG_CHARS = 500

logger = logging.getLogger(__name__)

__all__ = ["WebClient"]


class WebClient:
    """Async HTTP client for Open WebUI's model listing endpoint."""

    def __init__(self, base_url: str, token: str, *, request_timeout: int = 300) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(float(request_timeout), connect=10.0),
            headers={"Authorization": f"Bearer {token}"},
        )

    async def get_models(self) -> dict[str, Any]:
        logger.debug("GET /api/models → %s", self._client.base_url)
        resp = await self._client.get("/api/models")
        logger.debug("GET /api/models ← HTTP %s (%d bytes)", resp.status_code, len(resp.content))
        resp.raise_for_status()
        return resp.json()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> WebClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
