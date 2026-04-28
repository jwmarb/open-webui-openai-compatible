"""Orchestrator-style integration test: 8 parallel subagent tool calls.

Simulates an AI orchestrator dispatching 8 parallel tool calls to three
subagent types (explorer, docs, implementer) when asked to build a Tetris
game in React. Validates the proxy doesn't truncate or drop data during
high-fanout parallel tool calling.

Run with real credentials:
    OPEN_WEBUI_URL=https://... USER_TOKEN=<jwt> python -m pytest tests/integration/test_orchestrator_tool_calls.py -v
"""

import json

import pytest
from openai import OpenAI

from .conftest import skip_without_real_instance

pytestmark = skip_without_real_instance

SUBAGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "dispatch_explorer",
            "description": (
                "Dispatch an explorer subagent to analyze code, project structure, "
                "or architectural patterns. Returns a descriptive summary of findings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Unique identifier for this subagent task"},
                    "query": {"type": "string", "description": "What to explore or analyze"},
                    "scope": {
                        "type": "string",
                        "enum": ["architecture", "patterns", "state_management", "rendering", "testing"],
                        "description": "Area of exploration",
                    },
                },
                "required": ["task_id", "query", "scope"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_docs",
            "description": (
                "Dispatch a docs subagent to find and summarize external API "
                "documentation, library references, or best-practice guides."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Unique identifier for this subagent task"},
                    "topic": {"type": "string", "description": "API or library topic to research"},
                    "focus": {
                        "type": "string",
                        "enum": ["hooks", "rendering", "animation", "state", "components"],
                        "description": "Documentation focus area",
                    },
                },
                "required": ["task_id", "topic", "focus"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_implementer",
            "description": (
                "Dispatch an implementer subagent to write code for a specific "
                "module or component. Returns the generated source code."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string", "description": "Unique identifier for this subagent task"},
                    "component": {"type": "string", "description": "Name of the component or module to implement"},
                    "requirements": {"type": "string", "description": "Detailed implementation requirements"},
                    "dependencies": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Task IDs this implementation depends on",
                    },
                },
                "required": ["task_id", "component", "requirements"],
            },
        },
    },
]

ORCHESTRATOR_PROMPT = (
    "You are an orchestrator agent managing 8 subagents in parallel. "
    "You have three tool types: dispatch_explorer, dispatch_docs, and dispatch_implementer.\n\n"
    "Your task: plan and execute building a modern Tetris game using the latest version of React.\n\n"
    "You MUST dispatch EXACTLY 8 tool calls in parallel — all in a SINGLE response, "
    "not sequentially. Distribute them as follows:\n"
    "- 3x dispatch_explorer: (1) analyze React project structure & best practices, "
    "(2) analyze Tetris game logic patterns, (3) analyze state management for games\n"
    "- 2x dispatch_docs: (1) research React 19 hooks & APIs, "
    "(2) research Canvas/requestAnimationFrame for game rendering\n"
    "- 3x dispatch_implementer: (1) implement game board component, "
    "(2) implement tetromino logic & collision detection, "
    "(3) implement game controls & scoring system\n\n"
    "Give each call a unique task_id like 'explore-1', 'docs-1', 'impl-1', etc. "
    "Call all 8 tools at once in parallel."
)

VALID_TOOL_NAMES = {"dispatch_explorer", "dispatch_docs", "dispatch_implementer"}

_FAKE_RESULT = json.dumps({"status": "complete", "summary": "Task finished successfully."})


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


def _assemble_streaming_tool_calls(stream) -> tuple[dict[int, dict[str, str]], str | None]:
    tool_calls: dict[int, dict[str, str]] = {}
    finish_reason: str | None = None

    for chunk in stream:
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.delta.tool_calls:
            for tc in choice.delta.tool_calls:
                idx = tc.index
                if idx not in tool_calls:
                    tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                if tc.id:
                    tool_calls[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls[idx]["arguments"] += tc.function.arguments
        if choice.finish_reason:
            finish_reason = choice.finish_reason

    return tool_calls, finish_reason


def _build_roundtrip_messages(tool_calls_data) -> list[dict]:
    """Build turn-2 messages from turn-1 tool calls (SDK objects or assembled dicts)."""
    messages: list[dict] = [{"role": "user", "content": ORCHESTRATOR_PROMPT}]

    if hasattr(tool_calls_data[0], "id"):
        serialized = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls_data
        ]
        tool_ids = [(tc.id, tc.function.name) for tc in tool_calls_data]
    else:
        serialized = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls_data
        ]
        tool_ids = [(tc["id"], tc["name"]) for tc in tool_calls_data]

    messages.append({"role": "assistant", "tool_calls": serialized})
    for tc_id, _ in tool_ids:
        messages.append({"role": "tool", "tool_call_id": tc_id, "content": _FAKE_RESULT})

    return messages


class TestOrchestratorParallelToolCalls:

    def test_non_streaming_completes(self, openai_client, model_id):
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": ORCHESTRATOR_PROMPT}],
            tools=SUBAGENT_TOOLS,
        )
        assert resp.choices, "Response has no choices"
        choice = resp.choices[0]
        assert choice.finish_reason is not None, "Response ended without finish_reason"
        assert choice.message.tool_calls is not None, "Expected tool calls in response"
        assert len(choice.message.tool_calls) > 0, "Got empty tool_calls list"

        for tc in choice.message.tool_calls:
            assert tc.id, "Tool call missing id — response likely truncated"
            assert tc.function.name in VALID_TOOL_NAMES, f"Unknown tool: {tc.function.name}"
            json.loads(tc.function.arguments)  # raises on truncated JSON

    def test_streaming_completes(self, openai_client, model_id):
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": ORCHESTRATOR_PROMPT}],
            tools=SUBAGENT_TOOLS,
            stream=True,
        )
        tool_calls, finish_reason = _assemble_streaming_tool_calls(stream)

        assert finish_reason is not None, "Stream ended without finish_reason — likely truncated"
        assert len(tool_calls) > 0, "Stream produced zero tool calls"

        for idx, tc in tool_calls.items():
            assert tc["id"], f"Tool call at index {idx} missing id — stream truncated"
            assert tc["name"], f"Tool call at index {idx} missing name — stream truncated"
            json.loads(tc["arguments"])  # raises on truncated JSON

    def test_roundtrip_non_streaming(self, openai_client, model_id):
        # Turn 1: get tool calls
        resp = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": ORCHESTRATOR_PROMPT}],
            tools=SUBAGENT_TOOLS,
        )
        assert resp.choices[0].message.tool_calls, "Turn 1 produced no tool calls"

        # Turn 2: feed results back, expect a text answer
        messages = _build_roundtrip_messages(resp.choices[0].message.tool_calls)
        final = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=SUBAGENT_TOOLS,
        )
        assert final.choices, "Turn 2 response has no choices"
        assert final.choices[0].finish_reason is not None, "Turn 2 ended without finish_reason"
        assert final.choices[0].message.content, "Turn 2 produced no content — response truncated"

    def test_roundtrip_streaming(self, openai_client, model_id):
        # Turn 1: streaming tool calls
        stream = openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": ORCHESTRATOR_PROMPT}],
            tools=SUBAGENT_TOOLS,
            stream=True,
        )
        tool_calls, finish_reason = _assemble_streaming_tool_calls(stream)
        assert finish_reason is not None, "Turn 1 stream ended without finish_reason"
        assert len(tool_calls) > 0, "Turn 1 stream produced no tool calls"

        # Turn 2: stream the final synthesis
        ordered = [tool_calls[idx] for idx in sorted(tool_calls.keys())]
        messages = _build_roundtrip_messages(ordered)
        final_stream = openai_client.chat.completions.create(
            model=model_id,
            messages=messages,
            tools=SUBAGENT_TOOLS,
            stream=True,
        )

        content = ""
        final_finish_reason = None
        for chunk in final_stream:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices[0].finish_reason:
                final_finish_reason = chunk.choices[0].finish_reason

        assert final_finish_reason is not None, "Turn 2 stream ended without finish_reason"
        assert len(content) > 0, "Turn 2 stream produced no content — response truncated"
