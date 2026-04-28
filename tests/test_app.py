import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from src.main import _extract_finish_reason, app


class MockWebClient:
    # Matches spec of src.client.WebClient so patch("src.main.WebClient") replaces it transparently.
    # Real async methods (not AsyncMock) so lifespan's await aclose() works.

    def __init__(self, *_args, get_models=None, post_chat_completion=None, **_kwargs):
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
        http400 = httpx.Response(400, request=httpx.Request("POST", "/"))

        async def fake_post_stream(body, stream=False):
            async def failing_generator():
                raise httpx.HTTPStatusError(
                    "Bad Request",
                    request=httpx.Request("POST", "/"),
                    response=http400,
                )
                yield  # noqa: F841 — makes this an async generator

            return failing_generator()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.1",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 400
                body = response.json()
                assert body["error"]["type"] == "api_error"
                assert body["error"]["code"] == 400

    def test_chat_streaming_timeout_error(self):
        async def fake_post_stream(body, stream=False):
            async def failing_generator():
                raise httpx.ReadTimeout("timed out")
                yield  # noqa: F841

            return failing_generator()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "llama-3.1",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"
                assert body["error"]["code"] == 504

    # --- Mid-stream error paths (headers already sent → 200, error in SSE) ---

    def test_chat_streaming_mid_stream_connection_error(self):
        async def fake_post_stream(body, stream=False):
            async def partial_then_disconnect():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                raise httpx.RemoteProtocolError("connection reset")

            return partial_then_disconnect()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                error_payload = json.loads(lines[1].removeprefix("data: "))
                assert error_payload["error"]["type"] == "api_error"
                assert error_payload["error"]["code"] == 502
                assert lines[2] == "data: [DONE]"

    def test_chat_streaming_mid_stream_generic_error(self):
        async def fake_post_stream(body, stream=False):
            async def partial_then_fail():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                raise RuntimeError("unexpected")

            return partial_then_fail()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                error_payload = json.loads(lines[1].removeprefix("data: "))
                assert error_payload["error"]["type"] == "server_error"
                assert error_payload["error"]["code"] == 502
                assert lines[2] == "data: [DONE]"

    # --- finish_reason injection ---

    def test_streaming_injects_finish_reason_when_upstream_omits_it(self):
        async def fake_post_stream(body, stream=False):
            async def gen_no_finish_reason():
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                yield b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'

            return gen_no_finish_reason()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                # Last data line should be the injected finish_reason
                last_payload = json.loads(lines[-1].removeprefix("data: "))
                assert last_payload["choices"][0]["finish_reason"] == "stop"

    def test_streaming_does_not_inject_when_upstream_already_sends_finish_reason(self):
        async def fake_post_stream(body, stream=False):
            async def gen_with_finish_reason():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

            return gen_with_finish_reason()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                # Should only have the 2 upstream chunks, no injected third chunk
                assert len(lines) == 2

    # --- Non-streaming error paths ---

    def test_chat_non_streaming_connection_error(self):
        async def fake_post(body, stream=False):
            raise httpx.RemoteProtocolError("connection reset")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 502
                body = response.json()
                assert body["error"]["type"] == "api_error"
                assert body["error"]["code"] == 502

    def test_chat_non_streaming_generic_error(self):
        async def fake_post(body, stream=False):
            raise RuntimeError("unexpected")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 502
                body = response.json()
                assert body["error"]["type"] == "server_error"
                assert body["error"]["code"] == 502

    def test_chat_non_streaming_connect_timeout(self):
        async def fake_post(body, stream=False):
            raise httpx.ConnectTimeout("connect timed out")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"

    def test_chat_streaming_connect_timeout(self):
        async def fake_post_stream(body, stream=False):
            async def failing_gen():
                raise httpx.ConnectTimeout("connect timed out")
                yield  # noqa: F841

            return failing_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"

    # --- Thinking variant end-to-end ---

    def test_chat_thinking_variant_extended_e2e(self):
        captured = {}
        mock_result = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "opus",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

        async def fake_post(body, stream=False):
            captured.update(body)
            return mock_result

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "bedrock-claude-4-6-opus:extended", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 200
                assert captured["model"] == "bedrock-claude-4-6-opus"
                assert captured["thinking"]["type"] == "enabled"
                assert captured["thinking"]["budget_tokens"] == 32000
                assert captured["max_tokens"] >= 64000

    def test_chat_thinking_variant_adaptive_e2e(self):
        captured = {}
        mock_result = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "sonnet",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

        async def fake_post(body, stream=False):
            captured.update(body)
            return mock_result

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "bedrock-claude-4-5-sonnet:adaptive",
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                assert response.status_code == 200
                assert captured["model"] == "bedrock-claude-4-5-sonnet"
                assert captured["thinking"]["type"] == "adaptive"
                assert "budget_tokens" not in captured["thinking"]
                assert captured["max_tokens"] >= 64000

    # --- Request rewriting end-to-end ---

    def test_chat_passes_extra_fields_through(self):
        captured = {}
        mock_result = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        }

        async def fake_post(body, stream=False):
            captured.update(body)
            return mock_result

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "extra_body": {},
                        "api_base": "http://x",
                        "custom_llm_provider": "openai",
                    },
                )
                assert response.status_code == 200
                assert captured["model"] == "m"
                assert captured["extra_body"] == {}
                assert captured["api_base"] == "http://x"
                assert captured["custom_llm_provider"] == "openai"

    def test_chat_empty_body(self):
        captured = {}
        mock_result = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
        }

        async def fake_post(body, stream=False):
            captured.update(body)
            return mock_result

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post("/v1/chat/completions", json={})
                assert response.status_code == 200
                assert captured == {}


class TestModelsThinkingVariants:
    def test_models_with_claude_includes_thinking_variants(self):
        mock_raw = {
            "data": [
                {"id": "bedrock-claude-4-6-opus", "owned_by": "Anthropic", "created": 1700000000},
            ]
        }

        async def fake_get():
            return mock_raw

        mock = _make_mock_webclient(get_models=fake_get)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 200
                ids = [m["id"] for m in response.json()["data"]]
                assert ids == [
                    "bedrock-claude-4-6-opus",
                    "bedrock-claude-4-6-opus:extended",
                    "bedrock-claude-4-6-opus:adaptive",
                ]


class TestErrorResponseShape:
    def _assert_openai_error_shape(self, body: dict):
        assert set(body.keys()) == {"error"}, f"Expected only 'error' key, got {set(body.keys())}"
        err = body["error"]
        assert isinstance(err["message"], str) and len(err["message"]) > 0
        assert err["type"] in ("api_error", "server_error", "timeout_error", "invalid_request_error")
        assert err["code"] is None or isinstance(err["code"], int)

    def test_error_shape_http_status_errors(self):
        for status_code in (400, 401, 403, 429, 500, 502, 503):
            resp = httpx.Response(status_code, request=httpx.Request("POST", "/"))

            async def make_raiser(sc=status_code, r=resp):
                raise httpx.HTTPStatusError(f"Error {sc}", request=httpx.Request("POST", "/"), response=r)

            mock = _make_mock_webclient(post_chat_completion=lambda body, stream=False, _r=make_raiser: _r())

            with patch("src.main.WebClient", return_value=mock):
                with TestClient(app) as tc:
                    response = tc.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                    )
                    assert response.status_code == status_code, f"Expected {status_code}, got {response.status_code}"
                    self._assert_openai_error_shape(response.json())

    def test_error_shape_timeout(self):
        async def fake_post(body, stream=False):
            raise httpx.ReadTimeout("timed out")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_connection_error(self):
        async def fake_post(body, stream=False):
            raise httpx.RemoteProtocolError("reset")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_null_response(self):
        async def fake_post(body, stream=False):
            raise ValueError("Upstream returned empty or null response body")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_generic_error(self):
        async def fake_post(body, stream=False):
            raise RuntimeError("unexpected")

        mock = _make_mock_webclient(post_chat_completion=fake_post)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_models_upstream_error(self):
        http502 = httpx.Response(502, request=httpx.Request("GET", "/"))

        async def fake_get():
            raise httpx.HTTPStatusError("Bad Gateway", request=httpx.Request("GET", "/"), response=http502)

        mock = _make_mock_webclient(get_models=fake_get)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                self._assert_openai_error_shape(response.json())

    def test_error_shape_models_generic_error(self):
        async def fake_get():
            raise RuntimeError("connection refused")

        mock = _make_mock_webclient(get_models=fake_get)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                self._assert_openai_error_shape(response.json())

    def test_streaming_error_events_have_openai_shape(self):
        async def fake_post_stream(body, stream=False):
            async def gen():
                yield b'data: {"choices":[{"delta":{"content":"x"}}]}\n\n'
                raise httpx.ReadTimeout("timed out")

            return gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                error_payload = json.loads(lines[1].removeprefix("data: "))
                self._assert_openai_error_shape(error_payload)


class TestExtractFinishReason:
    def test_finish_reason_stop(self):
        chunk = b'data: {"choices":[{"finish_reason":"stop"}]}\n\n'
        assert _extract_finish_reason(chunk) == "stop"

    def test_finish_reason_tool_calls(self):
        chunk = b'data: {"choices":[{"finish_reason":"tool_calls"}]}\n\n'
        assert _extract_finish_reason(chunk) == "tool_calls"

    def test_finish_reason_null_returns_none(self):
        chunk = b'data: {"choices":[{"finish_reason":null}]}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_null_with_space_returns_none(self):
        chunk = b'data: {"choices":[{"finish_reason": null}]}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_absent_returns_none(self):
        chunk = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_done_marker_returns_none(self):
        chunk = b"data: [DONE]\n\n"
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_malformed_json(self):
        chunk = b'data: {malformed "finish_reason" json}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_no_needle_fast_path(self):
        chunk = b'data: {"choices":[{"delta":{"content":"hello world"}}]}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_multiple_choices_all_null(self):
        chunk = b'data: {"choices":[{"finish_reason":null},{"finish_reason":null}]}\n\n'
        assert _extract_finish_reason(chunk) is None

    def test_finish_reason_multiple_choices_non_null(self):
        chunk = b'data: {"choices":[{"finish_reason":"stop"},{"finish_reason":"length"}]}\n\n'
        assert _extract_finish_reason(chunk) == "stop"

        async def fake_post_null(body, stream=False):
            raise ValueError("Upstream returned empty or null response body")

        mock = _make_mock_webclient(post_chat_completion=fake_post_null)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "llama-3.1", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 502
                body = response.json()
                assert "error" in body
                assert body["error"]["type"] == "server_error"
                assert body["error"]["code"] == 502
                assert "empty response" in body["error"]["message"].lower()

    def test_chat_non_streaming_timeout(self):
        async def fake_post_timeout(body, stream=False):
            raise httpx.ReadTimeout("timed out")

        mock = _make_mock_webclient(post_chat_completion=fake_post_timeout)

        with patch("src.main.WebClient", return_value=mock):
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "llama-3.1", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"
                assert body["error"]["code"] == 504


class TestStreamEmptyRetry:
    def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        async def fake_post_stream(body, stream=False):
            nonlocal call_count
            call_count += 1

            if call_count == 1:

                async def empty_gen():
                    return
                    yield  # noqa: F841

                return empty_gen()

            async def real_gen():
                yield b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

            return real_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock), patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                assert len(lines) == 2
                assert call_count == 2

    def test_retry_exhausted_returns_injected_stop(self):
        call_count = 0

        async def fake_post_stream(body, stream=False):
            nonlocal call_count
            call_count += 1

            async def empty_gen():
                return
                yield  # noqa: F841

            return empty_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock), patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 2
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = [ln for ln in response.text.strip().split("\n") if ln.startswith("data: ")]
                last_payload = json.loads(lines[-1].removeprefix("data: "))
                assert last_payload["choices"][0]["finish_reason"] == "stop"
                expected_calls = 1 + mock_settings.stream_empty_retry_max
                assert call_count == expected_calls

    def test_retry_disabled_when_max_is_zero(self):
        call_count = 0

        async def fake_post_stream(body, stream=False):
            nonlocal call_count
            call_count += 1

            async def empty_gen():
                return
                yield  # noqa: F841

            return empty_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock), patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 0
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                assert call_count == 1

    def test_no_retry_when_first_chunk_has_data(self):
        call_count = 0

        async def fake_post_stream(body, stream=False):
            nonlocal call_count
            call_count += 1

            async def real_gen():
                yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
                yield b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'

            return real_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock), patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                assert call_count == 1

    def test_error_during_retry_is_not_retried(self):
        call_count = 0

        async def fake_post_stream(body, stream=False):
            nonlocal call_count
            call_count += 1

            if call_count == 1:

                async def empty_gen():
                    return
                    yield  # noqa: F841

                return empty_gen()

            async def error_gen():
                raise httpx.ReadTimeout("timed out")
                yield  # noqa: F841

            return error_gen()

        mock = _make_mock_webclient(post_chat_completion=fake_post_stream)

        with patch("src.main.WebClient", return_value=mock), patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 504
                assert call_count == 2
