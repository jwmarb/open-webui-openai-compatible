"""End-to-end integration tests against a real Open WebUI instance.

Run with real credentials:
    OPEN_WEBUI_URL=https://your-open-webui-instance.example.com USER_TOKEN=<jwt> python -m pytest tests/integration/ -v
"""

import json

from .conftest import skip_without_real_instance

pytestmark = skip_without_real_instance


class TestHealthIntegration:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsIntegration:

    def test_models_returns_openai_schema(self, client):
        """GET /v1/models should return translated OpenAI-compatible model list."""
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        body = resp.json()

        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert len(body["data"]) > 0, "Expected at least one model from upstream"

        for model in body["data"]:
            assert "id" in model
            assert model["object"] == "model"
            assert "created" in model
            assert "owned_by" in model
            assert "urlIdx" not in model  # leak check: upstream raw field must not appear
            assert "pipeline" not in model
            assert "tags" not in model
            assert "connection_type" not in model
            assert "info" not in model


class TestChatCompletionsIntegration:

    def _first_model_id(self, client) -> str:
        """Helper: fetch the first available model ID from /v1/models."""
        resp = client.get("/v1/models")
        models = resp.json()["data"]
        assert len(models) > 0, "No models available for chat test"
        return models[0]["id"]

    def test_chat_non_streaming(self, client):
        """POST /v1/chat/completions (non-streaming) returns OpenAI-compatible response."""
        model_id = self._first_model_id(client)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}],
                "stream": False,
            },
        )
        assert resp.status_code == 200
        body = resp.json()

        assert "id" in body
        assert body["object"] == "chat.completion"
        assert "choices" in body
        assert len(body["choices"]) > 0
        choice = body["choices"][0]
        assert "message" in choice
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert len(choice["message"]["content"]) > 0

    def test_chat_streaming(self, client):
        """POST /v1/chat/completions (streaming) returns SSE event stream."""
        model_id = self._first_model_id(client)
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": "Say 'hi' and nothing else."}],
                "stream": True,
            },
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")

            chunks = []
            for line in resp.iter_lines():
                line = line.strip()
                if not line:
                    continue
                chunks.append(line)

            assert len(chunks) > 0, "Expected SSE chunks from streaming response"
            data_lines = [c for c in chunks if c.startswith("data:")]
            assert len(data_lines) > 0, "Expected at least one 'data:' SSE line"

            first_data = json.loads(data_lines[0].removeprefix("data: "))
            assert "id" in first_data
            assert first_data["object"] == "chat.completion.chunk"
            assert "choices" in first_data
