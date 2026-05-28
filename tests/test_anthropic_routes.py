import json
from unittest.mock import patch

import httpx
import openai
from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from src.main import app

_DUMMY_REQUEST = httpx.Request("POST", "/")


def _completion(
    *,
    id: str = "chatcmpl-1",
    model: str = "m",
    content: str = "ok",
    finish_reason: str = "stop",
    created: int = 0,
    tool_calls: list | None = None,
    thinking_blocks: list | None = None,
) -> ChatCompletion:
    msg_kwargs: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg_kwargs["tool_calls"] = tool_calls
        msg_kwargs["content"] = None
    message = ChatCompletionMessage(**msg_kwargs)
    if thinking_blocks:
        message.thinking_blocks = thinking_blocks  # type: ignore[attr-defined]
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[CompletionChoice(
            index=0,
            message=message,
            finish_reason=finish_reason,
        )],
    )


def _chunk(
    *,
    id: str = "chatcmpl-1",
    model: str = "m",
    content: str | None = None,
    finish_reason: str | None = None,
    created: int = 0,
    reasoning_content: str | None = None,
    tool_calls: list | None = None,
) -> ChatCompletionChunk:
    delta_kwargs: dict = {}
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


def _parse_sse_events(response_text: str) -> list[dict]:
    events = []
    current_event_type = None
    for line in response_text.strip().split("\n"):
        if line.startswith("event: "):
            current_event_type = line[7:]
        elif line.startswith("data: ") and current_event_type:
            data = json.loads(line[6:])
            events.append({"event": current_event_type, "data": data})
            current_event_type = None
    return events


class MockChatCompletions:
    def __init__(self, handler):
        self._handler = handler

    async def create(self, **kwargs):
        return await self._handler(**kwargs)


class MockChat:
    def __init__(self, handler):
        self.completions = MockChatCompletions(handler)


class MockAsyncOpenAI:
    def __init__(self, handler=None, **_kwargs):
        self.chat = MockChat(handler or self._default_handler)

    @staticmethod
    async def _default_handler(**_kwargs):
        return _completion()

    async def close(self):
        pass


class MockWebClient:
    def __init__(self, *_args, **_kwargs):
        pass

    async def get_models(self) -> dict:
        return {"data": []}

    async def aclose(self) -> None:
        pass


def _patches(*, openai_handler=None):
    wc = MockWebClient()
    oa = MockAsyncOpenAI(handler=openai_handler)
    return (
        patch("src.main.WebClient", return_value=wc),
        patch("src.main.openai.AsyncOpenAI", return_value=oa),
    )


class TestMessagesNonStreaming:
    def test_basic_text_response(self):
        async def handler(**kwargs):
            return _completion(content="Hello!", model="claude-sonnet-4-20250514")

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert response.status_code == 200
                body = response.json()
                assert body["type"] == "message"
                assert body["role"] == "assistant"
                assert body["id"].startswith("msg_")
                assert body["content"][0]["type"] == "text"
                assert body["content"][0]["text"] == "Hello!"
                assert body["stop_reason"] == "end_turn"

    def test_tool_calls_response(self):
        from openai.types.chat.chat_completion_message_tool_call import (
            ChatCompletionMessageToolCall,
            Function,
        )

        async def handler(**kwargs):
            return _completion(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ChatCompletionMessageToolCall(
                        id="call_1",
                        type="function",
                        function=Function(name="search", arguments='{"q": "test"}'),
                    )
                ],
            )

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Search for test"}],
                        "tools": [{"name": "search", "input_schema": {"type": "object"}}],
                    },
                )
                assert response.status_code == 200
                body = response.json()
                assert body["stop_reason"] == "tool_use"
                tool_block = next(b for b in body["content"] if b["type"] == "tool_use")
                assert tool_block["name"] == "search"
                assert tool_block["input"] == {"q": "test"}

    def test_request_applies_rewrite_chat_body(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                extra = captured.get("extra_body", {})
                assert extra.get("chat_id", "").startswith("local:")

    def test_system_prompt_translation(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "system": "You are helpful.",
                    },
                )
                messages = captured.get("messages", [])
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "You are helpful."


class TestMessagesStreaming:
    def test_basic_streaming(self):
        async def handler(**kwargs):
            async def gen():
                yield _chunk(content="Hello")
                yield _chunk(content=" world")
                yield _chunk(finish_reason="stop")
            return gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "claude-sonnet-4-20250514",
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                events = _parse_sse_events(response.text)
                event_types = [e["event"] for e in events]
                assert "message_start" in event_types
                assert "content_block_start" in event_types
                assert "content_block_delta" in event_types
                assert "content_block_stop" in event_types
                assert "message_delta" in event_types
                assert "message_stop" in event_types

    def test_streaming_text_content(self):
        async def handler(**kwargs):
            async def gen():
                yield _chunk(content="Hi")
                yield _chunk(finish_reason="stop")
            return gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                events = _parse_sse_events(response.text)
                deltas = [e for e in events if e["event"] == "content_block_delta"]
                assert deltas[0]["data"]["delta"]["type"] == "text_delta"
                assert deltas[0]["data"]["delta"]["text"] == "Hi"

    def test_streaming_thinking_content(self):
        async def handler(**kwargs):
            async def gen():
                yield _chunk(reasoning_content="thinking...")
                yield _chunk(content="answer")
                yield _chunk(finish_reason="stop")
            return gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                events = _parse_sse_events(response.text)
                block_starts = [e for e in events if e["event"] == "content_block_start"]
                assert block_starts[0]["data"]["content_block"]["type"] == "thinking"
                assert block_starts[1]["data"]["content_block"]["type"] == "text"

    def test_streaming_finalize_on_missing_finish_reason(self):
        async def handler(**kwargs):
            async def gen():
                yield _chunk(content="hi")
            return gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                events = _parse_sse_events(response.text)
                event_types = [e["event"] for e in events]
                assert "message_stop" in event_types


class TestMessagesErrors:
    def test_upstream_error_returns_anthropic_format(self):
        async def handler(**kwargs):
            raise openai.APIStatusError(
                message="Service Unavailable",
                response=httpx.Response(503, request=_DUMMY_REQUEST),
                body=None,
            )

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert response.status_code == 503
                body = response.json()
                assert body["type"] == "error"
                assert body["error"]["type"] == "api_error"
                assert isinstance(body["error"]["message"], str)

    def test_timeout_error(self):
        async def handler(**kwargs):
            raise openai.APITimeoutError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert response.status_code == 504
                body = response.json()
                assert body["type"] == "error"
                assert body["error"]["type"] == "api_error"

    def test_connection_error(self):
        async def handler(**kwargs):
            raise openai.APIConnectionError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert response.status_code == 502
                body = response.json()
                assert body["type"] == "error"

    def test_streaming_first_chunk_error(self):
        async def handler(**kwargs):
            async def failing_gen():
                raise openai.APIStatusError(
                    message="Bad Request",
                    response=httpx.Response(400, request=_DUMMY_REQUEST),
                    body=None,
                )
                yield  # noqa: F841
            return failing_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 400
                body = response.json()
                assert body["type"] == "error"

    def test_streaming_mid_stream_error(self):
        async def handler(**kwargs):
            async def partial_then_fail():
                yield _chunk(content="hi")
                raise openai.APIConnectionError(request=_DUMMY_REQUEST)
            return partial_then_fail()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                events = _parse_sse_events(response.text)
                error_events = [e for e in events if e["event"] == "error"]
                assert len(error_events) >= 1
                assert error_events[0]["data"]["error"]["type"] == "api_error"
