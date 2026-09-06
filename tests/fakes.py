"""Shared upstream fakes.

One place for the fake clients every route test needs. Tests inject these via
`create_app(clients=...)` rather than patching module namespaces.
"""

from __future__ import annotations

from typing import Any

import httpx
import openai
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta
from openai.types.completion_usage import CompletionUsage

from src.main import UpstreamClients

__all__ = [
    "DUMMY_REQUEST",
    "FakeChatCompletions",
    "FakeOpenAI",
    "chunk",
    "completion",
    "fake_clients",
    "models_client",
    "parse_sse_events",
    "sse_data_lines",
    "usage_chunk",
]

DUMMY_REQUEST = httpx.Request("POST", "/")


def completion(
    *,
    id: str = "chatcmpl-1",
    model: str = "m",
    content: str | None = "Hello",
    finish_reason: str = "stop",
    created: int = 0,
    tool_calls: list | None = None,
    thinking_blocks: list | None = None,
) -> ChatCompletion:
    message = ChatCompletionMessage(role="assistant", content=content, tool_calls=tool_calls)
    if thinking_blocks is not None:
        message.thinking_blocks = thinking_blocks  # type: ignore[attr-defined]
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[CompletionChoice(index=0, message=message, finish_reason=finish_reason)],
    )


def chunk(
    *,
    id: str = "chatcmpl-1",
    model: str = "m",
    content: str | None = None,
    finish_reason: str | None = None,
    created: int = 0,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
) -> ChatCompletionChunk:
    delta_kwargs: dict[str, Any] = {}
    if content is not None:
        delta_kwargs["content"] = content
    if tool_calls is not None:
        delta_kwargs["tool_calls"] = tool_calls
    delta = ChoiceDelta(**delta_kwargs)
    if reasoning_content is not None:
        delta.reasoning_content = reasoning_content  # type: ignore[attr-defined]
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def usage_chunk(
    *,
    id: str = "chatcmpl-1",
    model: str = "m",
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=0,
        model=model,
        choices=[],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeChatCompletions:
    def __init__(self, handler=None) -> None:
        self._handler = handler

    async def create(self, **kwargs):
        if self._handler is None:
            return completion()
        return await self._handler(**kwargs)


class _FakeChat:
    def __init__(self, handler=None) -> None:
        self.completions = FakeChatCompletions(handler)


class FakeOpenAI:
    def __init__(self, handler=None) -> None:
        self.chat = _FakeChat(handler)

    async def close(self) -> None:
        return None


class _ModelsTransport(httpx.AsyncBaseTransport):
    """Serves /api/models from a payload or an async callable, with no network I/O.

    A callable may raise to simulate upstream failure; httpx errors propagate and
    anything else surfaces as a generic failure, matching real client behaviour.
    """

    def __init__(self, source: Any = None) -> None:
        self._source = source

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        source = self._source
        if source is None:
            return httpx.Response(200, json={"data": []}, request=request)
        if callable(source):
            payload = await source()
            return httpx.Response(200, json=payload, request=request)
        return httpx.Response(200, json=source, request=request)


def models_client(source: Any = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_ModelsTransport(source), base_url="http://upstream")


def fake_clients(
    *,
    openai_handler=None,
    models_payload: dict[str, Any] | None = None,
    models_source: Any = None,
) -> UpstreamClients:
    return UpstreamClients(
        models=models_client(models_source if models_source is not None else models_payload),
        chat=FakeOpenAI(openai_handler),  # type: ignore[arg-type]
    )


def sse_data_lines(response_text: str) -> list[str]:
    return [line[6:] for line in response_text.strip().split("\n") if line.startswith("data: ")]


def parse_sse_events(response_text: str) -> list[dict[str, Any]]:
    import json

    events: list[dict[str, Any]] = []
    current_event_type: str | None = None
    for line in response_text.strip().split("\n"):
        if line.startswith("event: "):
            current_event_type = line[7:]
        elif line.startswith("data: ") and current_event_type is not None:
            events.append({"event": current_event_type, "data": json.loads(line[6:])})
            current_event_type = None
    return events


def status_error(status_code: int, message: str = "error", body: Any = None) -> openai.APIStatusError:
    return openai.APIStatusError(
        message=message,
        response=httpx.Response(status_code, request=DUMMY_REQUEST),
        body=body,
    )
