"""OpenAI SDK integration tests against the proxy.

Verifies the proxy is fully compatible with the official OpenAI Python client
for models listing, non-streaming chat, streaming chat, and tool calls.

Run with real credentials:
    OPEN_WEBUI_URL=https://... USER_TOKEN=<jwt> python -m pytest tests/integration/test_openai_sdk.py -v
"""

import json

import pytest
from openai import OpenAI

from .conftest import skip_without_real_instance

pytestmark = skip_without_real_instance


@pytest.fixture
def openai_client(client):
    return OpenAI(
        base_url="http://testserver/v1",
        api_key="not-needed",
        http_client=client,
    )


@pytest.fixture
def model_id(openai_client):
    models = openai_client.models.list()
    assert len(models.data) > 0, "No models available from upstream"
    return models.data[0].id


class TestModelsSDK:

    def test_list_models(self, openai_client):
        models = openai_client.models.list()
        assert len(models.data) > 0
        for m in models.data:
            assert m.id
            assert m.object == "model"


class TestChatSDK:

    def test_non_streaming(self, openai_client, model_id):
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say exactly: hello"}],
        )
        assert resp.choices[0].message.role == "assistant"
        assert len(resp.choices[0].message.content) > 0
        assert resp.choices[0].finish_reason == "stop"

    def test_streaming(self, openai_client, model_id):
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Say exactly: hi"}],
            stream=True,
        )
        chunks = list(stream)
        assert len(chunks) > 0
        assert chunks[0].object == "chat.completion.chunk"

        content = ""
        for chunk in chunks:
            if chunk.choices and chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
        assert len(content) > 0

    def test_tool_call_non_streaming(self, openai_client, model_id):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=tools,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) > 0
        tc = choice.message.tool_calls[0]
        assert tc.function.name == "get_weather"
        args = json.loads(tc.function.arguments)
        assert "location" in args

    def test_tool_call_streaming(self, openai_client, model_id):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City name",
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=tools,
            stream=True,
        )

        tool_calls = {}
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"name": "", "arguments": ""}
                    if tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "tool_calls"
        assert len(tool_calls) > 0
        tc = tool_calls[0]
        assert tc["name"] == "get_weather"
        args = json.loads(tc["arguments"])
        assert "location" in args
