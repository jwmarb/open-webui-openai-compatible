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
) -> ChatCompletion:
    return ChatCompletion(
        id=id,
        object="chat.completion",
        created=created,
        model=model,
        choices=[CompletionChoice(
            index=0,
            message=ChatCompletionMessage(role="assistant", content=content),
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
) -> ChatCompletionChunk:
    delta = ChoiceDelta(content=content) if content is not None else ChoiceDelta()
    return ChatCompletionChunk(
        id=id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[ChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
    )


def _data_lines(response_text: str) -> list[str]:
    return [
        ln for ln in response_text.strip().split("\n")
        if ln.startswith("data: ") and ln != "data: [DONE]"
    ]


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
    def __init__(self, *_args, get_models=None, **_kwargs):
        self._get_models = get_models

    async def get_models(self) -> dict:
        return await self._get_models()

    async def aclose(self) -> None:
        pass


def _make_mock_webclient(get_models=None):
    return MockWebClient(get_models=get_models)


def _default_webclient():
    async def noop():
        return {"data": []}
    return _make_mock_webclient(get_models=noop)


def _patches(*, openai_handler=None, webclient=None):
    wc = webclient or _default_webclient()
    oa = MockAsyncOpenAI(handler=openai_handler)
    return (
        patch("src.main.WebClient", return_value=wc),
        patch("src.main.openai.AsyncOpenAI", return_value=oa),
    )


class TestHealth:
    def test_health(self):
        p_wc, p_oa = _patches()
        with p_wc, p_oa:
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

        wc = _make_mock_webclient(get_models=fake_get)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
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

        wc = _make_mock_webclient(get_models=fake_get_raises)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 502
                body = response.json()
                assert "error" in body
                assert body["error"]["type"] == "api_error"

    def test_models_generic_error(self):
        async def fake_get_raises():
            raise RuntimeError("connection refused")

        wc = _make_mock_webclient(get_models=fake_get_raises)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                assert response.status_code == 502
                body = response.json()
                assert body["error"]["type"] == "server_error"
                assert body["error"]["code"] == 502


class TestChatCompletionsEndpoint:
    def test_chat_non_streaming(self):
        async def handler(**kwargs):
            return _completion(id="chatcmpl-123", model="llama-3.1", content="Hello!", created=1700000000)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
                    "/v1/chat/completions",
                    json={"model": "llama-3.1", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 503
                body = response.json()
                assert "error" in body
                assert body["error"]["type"] == "api_error"
                assert body["error"]["code"] == 503

    def test_chat_streaming_upstream_error(self):
        async def handler(**kwargs):
            async def failing_generator():
                raise openai.APIStatusError(
                    message="Bad Request",
                    response=httpx.Response(400, request=_DUMMY_REQUEST),
                    body=None,
                )
                yield  # noqa: F841
            return failing_generator()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
        async def handler(**kwargs):
            async def failing_generator():
                raise openai.APITimeoutError(request=_DUMMY_REQUEST)
                yield  # noqa: F841
            return failing_generator()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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

    def test_chat_streaming_mid_stream_connection_error(self):
        async def handler(**kwargs):
            async def partial_then_disconnect():
                yield _chunk(content="hi")
                raise openai.APIConnectionError(request=_DUMMY_REQUEST)
            return partial_then_disconnect()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                error_payload = json.loads(lines[1].removeprefix("data: "))
                assert error_payload["error"]["type"] == "api_error"
                assert error_payload["error"]["code"] == 502

    def test_chat_streaming_mid_stream_generic_error(self):
        async def handler(**kwargs):
            async def partial_then_fail():
                yield _chunk(content="hi")
                raise RuntimeError("unexpected")
            return partial_then_fail()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                error_payload = json.loads(lines[1].removeprefix("data: "))
                assert error_payload["error"]["type"] == "server_error"
                assert error_payload["error"]["code"] == 502

    def test_streaming_injects_finish_reason_when_upstream_omits_it(self):
        async def handler(**kwargs):
            async def gen_no_finish_reason():
                yield _chunk(content="hello")
                yield _chunk(content=" world")
            return gen_no_finish_reason()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                last_payload = json.loads(lines[-1].removeprefix("data: "))
                assert last_payload["choices"][0]["finish_reason"] == "stop"

    def test_streaming_does_not_inject_when_upstream_already_sends_finish_reason(self):
        async def handler(**kwargs):
            async def gen_with_finish_reason():
                yield _chunk(content="hi")
                yield _chunk(finish_reason="stop")
            return gen_with_finish_reason()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                assert len(lines) == 2

    def test_chat_non_streaming_connection_error(self):
        async def handler(**kwargs):
            raise openai.APIConnectionError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
        async def handler(**kwargs):
            raise RuntimeError("unexpected")

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
        async def handler(**kwargs):
            raise openai.APITimeoutError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"

    def test_chat_streaming_connect_timeout(self):
        async def handler(**kwargs):
            async def failing_gen():
                raise openai.APITimeoutError(request=_DUMMY_REQUEST)
                yield  # noqa: F841
            return failing_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 504
                body = response.json()
                assert body["error"]["type"] == "timeout_error"

    def test_chat_thinking_variant_extended_e2e(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion(model="opus")

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "bedrock-claude-4-6-opus:extended", "messages": [{"role": "user", "content": "Hi"}]},
                )
                assert response.status_code == 200
                assert captured["model"] == "bedrock-claude-4-6-opus"
                extra = captured.get("extra_body", {})
                assert extra["thinking"]["type"] == "enabled"
                assert extra["thinking"]["budget_tokens"] == 32000
                assert captured["max_tokens"] >= 64000

    def test_chat_thinking_variant_adaptive_e2e(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion(model="sonnet")

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
                extra = captured.get("extra_body", {})
                assert extra["thinking"]["type"] == "adaptive"
                assert "budget_tokens" not in extra["thinking"]
                assert captured["max_tokens"] >= 64000

    def test_chat_passes_extra_fields_through(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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
                extra = captured.get("extra_body", {})
                assert "extra_body" in extra
                assert "api_base" in extra
                assert "custom_llm_provider" in extra

    def test_chat_empty_body(self):
        captured: dict = {}

        async def handler(**kwargs):
            captured.update(kwargs)
            return _completion()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post("/v1/chat/completions", json={})
                assert response.status_code == 200
                assert captured.get("extra_body") is None


class TestModelsThinkingVariants:
    def test_models_with_claude_includes_thinking_variants(self):
        mock_raw = {
            "data": [
                {"id": "bedrock-claude-4-6-opus", "owned_by": "Anthropic", "created": 1700000000},
            ]
        }

        async def fake_get():
            return mock_raw

        wc = _make_mock_webclient(get_models=fake_get)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
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
            async def make_raiser(sc=status_code, **kwargs):
                raise openai.APIStatusError(
                    message=f"Error {sc}",
                    response=httpx.Response(sc, request=_DUMMY_REQUEST),
                    body=None,
                )

            p_wc, p_oa = _patches(openai_handler=make_raiser)
            with p_wc, p_oa:
                with TestClient(app) as tc:
                    response = tc.post(
                        "/v1/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                    )
                    assert response.status_code == status_code, f"Expected {status_code}, got {response.status_code}"
                    self._assert_openai_error_shape(response.json())

    def test_error_shape_timeout(self):
        async def handler(**kwargs):
            raise openai.APITimeoutError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_connection_error(self):
        async def handler(**kwargs):
            raise openai.APIConnectionError(request=_DUMMY_REQUEST)

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                self._assert_openai_error_shape(response.json())

    def test_error_shape_generic_error(self):
        async def handler(**kwargs):
            raise RuntimeError("unexpected")

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
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

        wc = _make_mock_webclient(get_models=fake_get)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                self._assert_openai_error_shape(response.json())

    def test_error_shape_models_generic_error(self):
        async def fake_get():
            raise RuntimeError("connection refused")

        wc = _make_mock_webclient(get_models=fake_get)
        p_wc, p_oa = _patches(webclient=wc)

        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.get("/v1/models")
                self._assert_openai_error_shape(response.json())

    def test_streaming_error_events_have_openai_shape(self):
        async def handler(**kwargs):
            async def gen():
                yield _chunk(content="x")
                raise openai.APITimeoutError(request=_DUMMY_REQUEST)
            return gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa:
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                lines = _data_lines(response.text)
                error_payload = json.loads(lines[1].removeprefix("data: "))
                self._assert_openai_error_shape(error_payload)


class TestStreamEmptyRetry:
    def test_retry_succeeds_on_second_attempt(self):
        call_count = 0

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                async def empty_gen():
                    return
                    yield  # noqa: F841
                return empty_gen()

            async def real_gen():
                yield _chunk(content="hello")
                yield _chunk(finish_reason="stop")
            return real_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                assert len(lines) == 2
                assert call_count == 2

    def test_retry_exhausted_returns_injected_stop(self):
        call_count = 0

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1
            async def empty_gen():
                return
                yield  # noqa: F841
            return empty_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 2
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                lines = _data_lines(response.text)
                last_payload = json.loads(lines[-1].removeprefix("data: "))
                assert last_payload["choices"][0]["finish_reason"] == "stop"
                expected_calls = 1 + mock_settings.stream_empty_retry_max
                assert call_count == expected_calls

    def test_retry_disabled_when_max_is_zero(self):
        call_count = 0

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1
            async def empty_gen():
                return
                yield  # noqa: F841
            return empty_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
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

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1
            async def real_gen():
                yield _chunk(content="hi")
                yield _chunk(finish_reason="stop")
            return real_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 200
                assert call_count == 1

    def test_non_empty_stream_errors_are_also_retried(self):
        call_count = 0

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1
            async def error_gen():
                raise openai.APITimeoutError(request=_DUMMY_REQUEST)
                yield  # noqa: F841
            return error_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 504
                assert call_count == 4  # 1 initial + 3 retries

    def test_client_4xx_errors_are_not_retried(self):
        call_count = 0

        async def handler(**kwargs):
            nonlocal call_count
            call_count += 1

            async def error_gen():
                raise openai.NotFoundError(
                    message="Model does not exist",
                    response=httpx.Response(404, request=_DUMMY_REQUEST),
                    body=None,
                )
                yield  # noqa: F841
            return error_gen()

        p_wc, p_oa = _patches(openai_handler=handler)
        with p_wc, p_oa, patch("src.main.settings") as mock_settings:
            mock_settings.stream_empty_retry_max = 3
            mock_settings.log_level = "WARNING"
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "nonexistent", "messages": [{"role": "user", "content": "Hi"}], "stream": True},
                )
                assert response.status_code == 404
                assert call_count == 1
