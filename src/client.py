from __future__ import annotations

from typing import AsyncIterator

import httpx

from .settings import settings


class WebClient:
    """Async HTTP client for Open WebUI upstream API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=settings.open_webui_url,
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={"Authorization": f"Bearer {settings.user_token}"},
        )

    async def get_models(self) -> dict:
        """GET /api/models — fetch available models from Open WebUI."""
        resp = await self._client.get("/api/models")
        resp.raise_for_status()
        return resp.json()

    async def post_chat_completion(
        self, body: dict, *, stream: bool = False,
    ) -> dict | AsyncIterator[str]:
        """POST /api/chat/completions — JWT auth via Bearer header."""
        if stream:
            return self._stream_chat(body)
        resp = await self._client.post("/api/chat/completions", json=body)
        resp.raise_for_status()
        return resp.json()

    async def _stream_chat(self, body: dict) -> AsyncIterator[str]:
        """Stream SSE response from /api/chat/completions."""
        async with self._client.stream(
            "POST", "/api/chat/completions", json=body,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                yield line

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
