"""Integration tests for parallel tool calling through the proxy.

Verifies that the proxy correctly handles requests where the model returns
multiple tool calls in a single response (parallel tool calling).

Run with real credentials:
    OPEN_WEBUI_URL=https://... USER_TOKEN=<jwt> python -m pytest tests/integration/test_parallel_tool_calls.py -v
"""

import json

import pytest
from openai import OpenAI

from .conftest import skip_without_real_instance

pytestmark = skip_without_real_instance

TOOLS = [
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
                        "description": "City name, e.g. 'Tokyo'",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local time for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo'",
                    },
                },
                "required": ["location"],
            },
        },
    },
]

PROMPT = (
    "What is the weather AND the current time in both Tokyo and Paris? "
    "You MUST call get_weather and get_time for BOTH cities in parallel. "
    "Call all four tool calls at once."
)


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


class TestParallelToolCalls:

    def test_parallel_tool_calls_non_streaming(self, openai_client, model_id):
        """Model should return multiple tool calls in a single response."""
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            tools=TOOLS,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 2, (
            f"Expected multiple parallel tool calls, got {len(choice.message.tool_calls)}"
        )

        called_functions = {tc.function.name for tc in choice.message.tool_calls}
        assert called_functions & {"get_weather", "get_time"}, (
            f"Expected at least get_weather or get_time, got {called_functions}"
        )

        for tc in choice.message.tool_calls:
            assert tc.id, "Tool call must have an id"
            assert tc.type == "function"
            args = json.loads(tc.function.arguments)
            assert "location" in args

    def test_parallel_tool_calls_streaming(self, openai_client, model_id):
        """Streaming should correctly assemble multiple parallel tool calls from deltas."""
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            tools=TOOLS,
            stream=True,
        )

        tool_calls: dict[int, dict[str, str]] = {}
        finish_reason = None
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function and tc.function.name:
                        tool_calls[idx]["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        tool_calls[idx]["arguments"] += tc.function.arguments
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "tool_calls"
        assert len(tool_calls) >= 2, (
            f"Expected multiple parallel tool calls, got {len(tool_calls)}"
        )

        called_functions = {tc["name"] for tc in tool_calls.values()}
        assert called_functions & {"get_weather", "get_time"}, (
            f"Expected at least get_weather or get_time, got {called_functions}"
        )

        for tc in tool_calls.values():
            assert tc["id"], "Tool call must have an id"
            assert tc["name"], "Tool call must have a function name"
            args = json.loads(tc["arguments"])
            assert "location" in args

    def test_parallel_tool_roundtrip_non_streaming(self, openai_client, model_id):
        """Full roundtrip: get tool calls, send results back, get final answer."""
        # Step 1: Initial request — model should return tool calls
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": PROMPT}],
            tools=TOOLS,
        )
        choice = resp.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) >= 2

        # Step 2: Build follow-up with tool results
        messages: list[dict] = [
            {"role": "user", "content": PROMPT},
            {"role": "assistant", "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in choice.message.tool_calls
            ]},
        ]

        fake_results = {
            "get_weather": '{"temperature": "22°C", "condition": "sunny"}',
            "get_time": '{"time": "14:30 JST"}',
        }
        for tc in choice.message.tool_calls:
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": fake_results.get(tc.function.name, '{"result": "ok"}'),
            })

        # Step 3: Send tool results back — model should produce a text answer
        final = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=TOOLS,
        )
        final_choice = final.choices[0]
        assert final_choice.message.content is not None
        assert len(final_choice.message.content) > 0
