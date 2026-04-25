"""Async HTTP client for Open WebUI upstream API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, overload

import httpx

_MAX_ERROR_LOG_CHARS = 500

logger = logging.getLogger(__name__)

__all__ = ["WebClient"]


_DEFAULT_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_STREAM_TIMEOUT = httpx.Timeout(None, connect=10.0, pool=120.0)


class WebClient:
    """Async HTTP client wrapping Open WebUI's internal API endpoints."""

    def __init__(self, base_url: str, token: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=_DEFAULT_TIMEOUT,
            headers={"Authorization": f"Bearer {token}"},
        )

    async def get_models(self) -> dict[str, Any]:
        """GET /api/models — fetch available models from Open WebUI."""
        resp = await self._client.get("/api/models")
        resp.raise_for_status()
        return resp.json()

    @overload
    async def post_chat_completion(
        self, body: dict[str, Any], *, stream: bool = False,
    ) -> dict[str, Any]: ...

    @overload
    async def post_chat_completion(
        self, body: dict[str, Any], *, stream: bool = True,
    ) -> AsyncIterator[bytes]: ...

    async def post_chat_completion(
        self, body: dict[str, Any], *, stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """POST /api/chat/completions — JWT auth via Bearer header.

        When stream=True, returns an async iterator of raw bytes preserving
        the upstream SSE framing exactly as received.
        """
        if stream:
            return self._stream_chat_raw(body)
        resp = await self._client.post("/api/chat/completions", json=body)
        if resp.is_error:
            logger.error(
                "Upstream %s: %s",
                resp.status_code,
                resp.text[:_MAX_ERROR_LOG_CHARS],
            )
        resp.raise_for_status()
        return resp.json()

    async def _stream_chat_raw(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Yield raw byte chunks from upstream, preserving SSE framing exactly."""
        async with self._client.stream(
            "POST", "/api/chat/completions", json=body, timeout=_STREAM_TIMEOUT,
        ) as resp:
            if resp.is_error:
                error_body = await resp.aread()
                logger.error(
                    "Upstream stream %s: %s",
                    resp.status_code,
                    error_body[:_MAX_ERROR_LOG_CHARS],
                )
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
