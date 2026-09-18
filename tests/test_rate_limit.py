"""Tests for the upstream rate-limit stall (ADR-0006).

Covers detection and reset parsing, the per-request stall budget, and all
four pre-header route seams — both frontends, streaming and non-streaming —
each with a recover case (final 200 after stalling) and an exhaustion case
(429 + Retry-After). Real short sleeps are used so the deadline arithmetic
stays honest; the whole file runs in a few seconds.
"""

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from fastapi.testclient import TestClient
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as CompletionChoice

from src.main import create_app
from src.open_webui.rate_limit import RateLimitStall, is_rate_limit, seconds_until_reset
from tests.fakes import chunk, completion, fake_clients, parse_sse_events, status_error

OPENAI_SETTINGS = "src.proxy.openai.routes.settings"
ANTHROPIC_SETTINGS = "src.proxy.anthropic.routes.settings"


def _detail(reset_in_seconds: float) -> str:
    reset_at = datetime.now(UTC) + timedelta(seconds=reset_in_seconds)
    return (
        "Rate limit exceeded for end_user: test@example.com. "
        "Limit type: requests. Current limit: 20, Remaining: 0. "
        f"Limit resets at: {reset_at.strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _rate_limit_400(reset_in_seconds: float) -> Exception:
    return status_error(400, "rate limited", body={"detail": _detail(reset_in_seconds)})

def _rate_limit_429() -> Exception:
    return status_error(429, "Too Many Requests")


def _anthropic_completion(content: str = "ok") -> ChatCompletion:
    """A completion with no tool_calls field, the upstream shape when none are made."""
    message = ChatCompletionMessage(role="assistant", content=content)
    return ChatCompletion(
        id="chatcmpl-1",
        object="chat.completion",
        created=0,
        model="m",
        choices=[CompletionChoice(index=0, message=message, finish_reason="stop")],
    )


class _App:
    """Context manager that builds the app with a scripted upstream handler."""

    def __init__(self, handler) -> None:
        self._handler = handler

    def __enter__(self):
        return create_app(clients=fake_clients(openai_handler=self._handler))

    def __exit__(self, *exc):
        return False


class _FirstChunkRateLimitedStream:
    """A stream whose first-chunk pre-read raises the given exception."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self._exc


def _openai_content_lines(response_text: str) -> list[str | None]:
    contents = []
    for line in response_text.strip().split("\n"):
        if line.startswith("data: ") and line != "data: [DONE]":
            payload = json.loads(line[6:])
            choices = payload.get("choices") or []
            contents.append(choices[0]["delta"].get("content") if choices else None)
    return contents


class TestDetection:
    def test_429_is_a_rate_limit(self):
        assert is_rate_limit(_rate_limit_429())

    def test_400_with_detail_is_a_rate_limit(self):
        assert is_rate_limit(_rate_limit_400(5.0))

    def test_bare_400_is_not_a_rate_limit(self):
        exc = status_error(400, "bad", body={"detail": "unknown_parameter: 'thinking'"})
        assert not is_rate_limit(exc)

    def test_500_with_rate_limit_detail_is_not(self):
        assert not is_rate_limit(status_error(500, "boom", body={"detail": _detail(5.0)}))

    def test_non_status_error_is_not(self):
        assert not is_rate_limit(RuntimeError("nope"))

    def test_seconds_until_reset(self):
        reset_in = seconds_until_reset(_rate_limit_400(5.0))
        assert reset_in is not None
        # The detail timestamp is truncated to whole seconds, so the parsed
        # value can be up to 1 s short of the true offset.
        assert 4.0 < reset_in <= 5.0

    def test_seconds_until_reset_missing(self):
        assert seconds_until_reset(_rate_limit_429()) is None

    def test_live_gateway_format_with_utc_suffix(self):
        """Pin the exact detail format emitted by genai.arizona.edu (verified 2026-09-18).

        The timestamp carries a trailing ``UTC`` token that must not be
        swallowed by the parser.
        """
        target = datetime(2026, 9, 19, tzinfo=UTC)
        detail = (
            "Rate limit exceeded for end_user: test@example.com. "
            "Limit type: requests. Current limit: 20, Remaining: 0. "
            f"Limit resets at: {target.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )
        exc = status_error(400, "rate limited", body={"detail": detail})
        assert is_rate_limit(exc)
        reset_in = seconds_until_reset(exc)
        assert reset_in is not None
        expected = (target - datetime.now(UTC)).total_seconds()
        assert abs(reset_in - expected) < 1.0

class TestStallBudget:
    def test_disabled_budget_never_stalls(self):
        stall = RateLimitStall(0)
        assert stall.sleep_for(_rate_limit_400(5.0)) is None
        assert not stall.exhausted

    def test_budget_spent_marks_exhausted(self):
        stall = RateLimitStall(0.05)
        assert stall.sleep_for(_rate_limit_400(5.0)) is not None
        time.sleep(0.06)
        assert stall.sleep_for(_rate_limit_400(5.0)) is None
        assert stall.exhausted

    def test_reset_in_past_retries_immediately(self):
        stall = RateLimitStall(5.0)
        assert stall.sleep_for(_rate_limit_400(-1.0)) == 0.0

    def test_unparseable_reset_falls_back_to_one_second(self):
        stall = RateLimitStall(5.0)
        assert stall.sleep_for(_rate_limit_429()) == 1.0

    def test_retry_after_ceil(self):
        # Snap the reset to an exact integer second so the ceil is deterministic.
        target = int(datetime.now(UTC).timestamp()) + 2
        ts = datetime.fromtimestamp(target, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")
        exc = status_error(
            400, "rate limited", body={"detail": f"Rate limit exceeded. Limit resets at: {ts}"}
        )
        stall = RateLimitStall(5.0)
        stall.sleep_for(exc)
        assert stall.retry_after() == 2
        stall2 = RateLimitStall(5.0)
        stall2.sleep_for(_rate_limit_429())
        assert stall2.retry_after() == 1


class TestOpenAISeams:
    def test_non_streaming_recovers(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _rate_limit_400(1.0)
            return completion(content="ok")

        with _App(handler) as app, patch(OPENAI_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 300
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )

        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ok"
        assert calls["n"] == 2

    def test_non_streaming_exhausts_to_429(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            raise _rate_limit_400(3.0)

        with _App(handler) as app, patch(OPENAI_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 1
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                started = time.monotonic()
                response = tc.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )
                elapsed = time.monotonic() - started

        assert response.status_code == 429
        assert 1 <= int(response.headers["retry-after"]) <= 3
        body = response.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in body["error"]["message"]
        assert 2 <= calls["n"] <= 4
        assert elapsed < 2.5

    def test_streaming_first_chunk_recovers(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FirstChunkRateLimitedStream(_rate_limit_400(1.0))

            async def gen():
                yield chunk(content="ok")
                yield chunk(finish_reason="stop")

            return gen()

        with _App(handler) as app, patch(OPENAI_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 300
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )

        assert response.status_code == 200
        assert [c for c in _openai_content_lines(response.text) if c] == ["ok"]
        assert calls["n"] == 2

    def test_streaming_create_exhausts_to_429(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            raise _rate_limit_400(3.0)

        with _App(handler) as app, patch(OPENAI_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 1
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                started = time.monotonic()
                response = tc.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                elapsed = time.monotonic() - started

        assert response.status_code == 429
        assert 1 <= int(response.headers["retry-after"]) <= 3
        body = response.json()
        assert body["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in body["error"]["message"]
        assert 2 <= calls["n"] <= 4
        assert elapsed < 2.5


class TestAnthropicSeams:
    def test_non_streaming_recovers(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _rate_limit_400(1.0)
            return _anthropic_completion()

        with _App(handler) as app, patch(ANTHROPIC_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 300
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )

        assert response.status_code == 200
        assert response.json()["content"][0]["text"] == "ok"
        assert calls["n"] == 2

    def test_non_streaming_exhausts_to_429(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            raise _rate_limit_400(3.0)

        with _App(handler) as app, patch(ANTHROPIC_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 1
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                started = time.monotonic()
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Hi"}],
                    },
                )
                elapsed = time.monotonic() - started

        assert response.status_code == 429
        assert 1 <= int(response.headers["retry-after"]) <= 3
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in body["error"]["message"]
        assert 2 <= calls["n"] <= 4
        assert elapsed < 2.5

    def test_streaming_first_chunk_recovers(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FirstChunkRateLimitedStream(_rate_limit_400(1.0))

            async def gen():
                yield chunk(content="ok")
                yield chunk(finish_reason="stop")

            return gen()

        with _App(handler) as app, patch(ANTHROPIC_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 300
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )

        assert response.status_code == 200
        events = parse_sse_events(response.text)
        types = [e["event"] for e in events]
        assert types.count("message_start") == 1
        assert types.count("message_stop") == 1
        text = "".join(
            e["data"].get("delta", {}).get("text", "")
            for e in events
            if e["event"] == "content_block_delta"
        )
        assert text == "ok"
        assert calls["n"] == 2

    def test_streaming_create_exhausts_to_429(self):
        calls = {"n": 0}

        async def handler(**kwargs):
            calls["n"] += 1
            raise _rate_limit_400(3.0)

        with _App(handler) as app, patch(ANTHROPIC_SETTINGS) as mock_settings:
            mock_settings.rate_limit_stall_max_seconds = 1
            mock_settings.stream_empty_retry_max = 0
            with TestClient(app) as tc:
                started = time.monotonic()
                response = tc.post(
                    "/v1/messages",
                    json={
                        "model": "m",
                        "max_tokens": 8,
                        "messages": [{"role": "user", "content": "Hi"}],
                        "stream": True,
                    },
                )
                elapsed = time.monotonic() - started

        assert response.status_code == 429
        assert 1 <= int(response.headers["retry-after"]) <= 3
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"
        assert "Rate limit exceeded" in body["error"]["message"]
        assert 2 <= calls["n"] <= 4
        assert elapsed < 2.5
