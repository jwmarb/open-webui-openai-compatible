"""Async HTTP client for Open WebUI upstream API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any, overload

import httpx

_MAX_ERROR_LOG_CHARS = 500

logger = logging.getLogger(__name__)

__all__ = ["WebClient"]


class WebClient:
    """Async HTTP client wrapping Open WebUI's internal API endpoints."""

    def __init__(self, base_url: str, token: str, *, request_timeout: int = 300) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=httpx.Timeout(float(request_timeout), connect=10.0),
            headers={"Authorization": f"Bearer {token}"},
        )
        self._stream_timeout = httpx.Timeout(None, connect=10.0, pool=float(request_timeout))

    async def get_models(self) -> dict[str, Any]:
        """GET /api/models — fetch available models from Open WebUI."""
        logger.debug("GET /api/models → %s", self._client.base_url)
        resp = await self._client.get("/api/models")
        logger.debug("GET /api/models ← HTTP %s (%d bytes)", resp.status_code, len(resp.content))
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
        logger.debug("POST /api/chat/completions → model=%s", body.get("model", "?"))
        resp = await self._client.post("/api/chat/completions", json=body)
        logger.debug(
            "POST /api/chat/completions ← HTTP %s (%d bytes)",
            resp.status_code,
            len(resp.content),
        )
        if resp.is_error:
            logger.error(
                "Upstream %s: %s",
                resp.status_code,
                resp.text[:_MAX_ERROR_LOG_CHARS],
            )
        resp.raise_for_status()
        result = resp.json()
        if result is None:
            logger.error("Upstream returned null body (HTTP %s)", resp.status_code)
            raise ValueError("Upstream returned empty or null response body")
        return result

    async def _stream_chat_raw(self, body: dict[str, Any]) -> AsyncIterator[bytes]:
        """Yield raw byte chunks from upstream, preserving SSE framing exactly."""
        logger.debug("POST /api/chat/completions (stream) → model=%s", body.get("model", "?"))
        async with self._client.stream(
            "POST", "/api/chat/completions", json=body, timeout=self._stream_timeout,
        ) as resp:
            logger.debug("POST /api/chat/completions (stream) ← HTTP %s", resp.status_code)
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
