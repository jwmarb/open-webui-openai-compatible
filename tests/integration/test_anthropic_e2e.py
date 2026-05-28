"""End-to-end integration tests for Anthropic Messages API against a real Open WebUI instance.

Run with real credentials:
    OPEN_WEBUI_URL=https://your-open-webui-instance.example.com USER_TOKEN=<jwt> python -m pytest tests/integration/ -v
"""

import json

from .conftest import skip_without_real_instance

pytestmark = skip_without_real_instance


class TestAnthropicMessagesIntegration:

    def _first_claude_model(self, client) -> str:
        resp = client.get("/v1/models")
        models = resp.json()["data"]
        claude_models = [m["id"] for m in models if "claude" in m["id"].lower() and ":" not in m["id"]]
        assert len(claude_models) > 0, "No Claude models available for Anthropic test"
        return claude_models[0]

    def test_messages_non_streaming(self, client):
        model_id = self._first_claude_model(client)
        resp = client.post(
            "/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
            },
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        body = resp.json()

        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert "id" in body
        assert body["model"] == model_id
        assert isinstance(body["content"], list)
        assert len(body["content"]) > 0
        assert body["content"][0]["type"] == "text"
        assert isinstance(body["content"][0]["text"], str)
        assert len(body["content"][0]["text"]) > 0
        assert body["stop_reason"] in ("end_turn", "max_tokens")
        assert "usage" in body
        assert "input_tokens" in body["usage"]
        assert "output_tokens" in body["usage"]

    def test_messages_streaming(self, client):
        model_id = self._first_claude_model(client)
        with client.stream(
            "POST",
            "/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "Say 'hi' and nothing else."}],
                "stream": True,
            },
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            events = []
            for line in resp.iter_lines():
                line = line.strip()
                if line.startswith("event: "):
                    events.append({"event": line.removeprefix("event: ")})
                elif line.startswith("data: ") and events:
                    events[-1]["data"] = json.loads(line.removeprefix("data: "))

            assert len(events) > 0
            assert events[0]["event"] == "message_start"
            assert events[0]["data"]["type"] == "message_start"
            assert events[-1]["event"] == "message_stop"

            event_types = [e["event"] for e in events]
            assert "content_block_start" in event_types
            assert "content_block_delta" in event_types
            assert "content_block_stop" in event_types
            assert "message_delta" in event_types

    def test_messages_with_system_prompt(self, client):
        model_id = self._first_claude_model(client)
        resp = client.post(
            "/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 50,
                "system": "You only respond with the word 'pineapple'.",
                "messages": [{"role": "user", "content": "What is your favorite fruit?"}],
            },
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"
        assert len(body["content"]) > 0

    def test_messages_with_thinking(self, client):
        model_id = self._first_claude_model(client)
        resp = client.post(
            "/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 16000,
                "thinking": {"type": "enabled", "budget_tokens": 5000},
                "messages": [{"role": "user", "content": "What is 15 * 37?"}],
            },
            headers={"x-api-key": "dummy", "anthropic-version": "2023-06-01"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "message"

        content_types = [block["type"] for block in body["content"]]
        assert "thinking" in content_types, "Expected thinking block in response"
        assert "text" in content_types, "Expected text block in response"

        thinking_block = next(b for b in body["content"] if b["type"] == "thinking")
        assert "thinking" in thinking_block
        assert "signature" in thinking_block
        assert len(thinking_block["signature"]) > 0
