import httpx
from unittest.mock import patch

from src.main import app


class MockWebClient:
    # Matches spec of src.client.WebClient so patch("src.main.WebClient") replaces it transparently.
    # Real async methods (not AsyncMock) so lifespan's await aclose() works.

    def __init__(self, get_models=None, post_chat_completion=None):
        self._get_models = get_models
        self._post_chat_completion = post_chat_completion

    async def get_models(self) -> dict:
        return await self._get_models()

    async def post_chat_completion(self, body: dict, *, stream: bool = False):
        return await self._post_chat_completion(body, stream=stream)

    async def aclose(self) -> None:
        pass


def _make_mock_webclient(get_models=None, post_chat_completion=None):
    return MockWebClient(get_models=get_models, post_chat_completion=post_chat_completion)


class TestHealth:

    def test_health(self):
        mock = _make_mock_webclient()
        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.get("/health")
                assert response.status_code == 200
                assert response.json() == {"status": "ok"}


class TestModelsEndpoint:

    def test_models_success(self):
        mock_raw = {
            "data": [
                {"id": "llama-3.1", "owned_by": "Meta", "created": 1700000000},
            ]
        }

        async def fake_get():
            return mock_raw

        mock = _make_mock_webclient(get_models=fake_get)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 200
                body = response.json()
                assert body["object"] == "list"
                assert len(body["data"]) == 1
                assert body["data"][0]["id"] == "llama-3.1"
                assert body["data"][0]["object"] == "model"

    def test_models_upstream_error(self):
        http502 = httpx.Response(502, request=httpx.Request("GET", "/"))

        async def fake_get_raises():
            raise httpx.HTTPStatusError("Bad Gateway", request=httpx.Request("GET", "/"), response=http502)

        mock = _make_mock_webclient(get_models=fake_get_raises)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 502
                body = response.json()
                assert "error" in body
                assert body["error"]["type"] == "api_error"

    def test_models_generic_error(self):
        async def fake_get_raises():
            raise RuntimeError("connection refused")

        mock = _make_mock_webclient(get_models=fake_get_raises)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 502
                body = response.json()
                assert body["error"]["type"] == "server_error"
                assert body["error"]["code"] == 502


class TestChatCompletionsEndpoint:

    def test_chat_non_streaming(self):
        mock_result = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1700000000,
            "model": "llama-3.1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
        }

        async def fake_post(body, stream=False):
            return mock_result

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "llama-3.1", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 200
                body = response.json()
                assert body["id"] == "chatcmpl-123"
                assert body["object"] == "chat.completion"

    def test_chat_upstream_error(self):
        http503 = httpx.Response(503, request=httpx.Request("POST", "/"))

        async def fake_post_raises(body, stream=False):
            raise httpx.HTTPStatusError("Service Unavailable", request=httpx.Request("POST", "/"), response=http503)

        mock = _make_mock_webclient(post_chat_completion=fake_post_raises)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "llama-3.1", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 503
                body = response.json()
                assert "error" in body
                assert body["error"]["type"] == "api_error"
                assert body["error"]["code"] == 503

    def test_chat_streaming_upstream_error(self):
        """Upstream 400 during streaming emits SSE error event instead of crashing."""
        import json
        http400 = httpx.Response(400, request=httpx.Request("POST", "/"))

        async def fake_post_stream(body, stream=False):
            async def failing_generator():
                raise httpx.HTTPStatusError(
                    "Bad Request",
                    request=httpx.Request("POST", "/"),
                    response=http400,
                )
                yield  # noqa: unreachable — makes this an async generator

            return failing_generator()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.1",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200  # headers already sent
                lines = [l for l in response.text.strip().split("\n") if l.startswith("data: ")]
                assert len(lines) == 2
                error_payload = json.loads(lines[0].removeprefix("data: "))
                assert error_payload["error"]["type"] == "api_error"
                assert error_payload["error"]["code"] == 400
                assert lines[1] == "data: [DONE]"

    def test_chat_streaming_generic_error(self):
        """Generic exception during streaming emits SSE error event."""
        import json

        async def fake_post_stream(body, stream=False):
            async def failing_generator():
                raise RuntimeError("connection reset")
                yield  # noqa: unreachable

            return failing_generator()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            from fastapi.testclient import TestClient
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.1",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 200
                lines = [l for l in response.text.strip().split("\n") if l.startswith("data: ")]
                assert len(lines) == 2
                error_payload = json.loads(lines[0].removeprefix("data: "))
                assert error_payload["error"]["type"] == "server_error"
                assert error_payload["error"]["code"] == 502
                assert lines[1] == "data: [DONE]"
