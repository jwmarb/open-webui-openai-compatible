"""Tests for proxy refresh trigger logic — 401 detection, sidecar spawn, lock."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import jwt
import openai
import pytest
from fastapi.testclient import TestClient

from src.auth import should_refresh
from src.main import create_app
from tests.fakes import fake_clients


def _app_with(handler):
    return create_app(clients=fake_clients(openai_handler=handler))


def _make_jwt(exp_offset_seconds: int) -> str:
    now = int(time.time())
    payload = {"id": "test-user", "iat": now, "exp": now + exp_offset_seconds}
    return jwt.encode(payload, "test-secret", algorithm="HS256")


@pytest.fixture
def token_file(tmp_path: Path) -> Path:
    return tmp_path / "token.json"


@pytest.fixture
def mock_settings_env(monkeypatch: pytest.MonkeyPatch, token_file: Path):
    monkeypatch.setenv("TOKEN_FILE", str(token_file))
    monkeypatch.delenv("USER_TOKEN", raising=False)


@pytest.fixture
def _setup_token(token_file: Path):
    def _set(token: str, expires_at: int | None = None):
        data = {"token": token}
        if expires_at is not None:
            data["expires_at"] = expires_at
        token_file.write_text(json.dumps(data))
    return _set


class TestRefreshTriggerOn401:
    """Tests for proxy behavior when upstream returns 401."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_settings_env: None, _setup_token):
        _setup_token("test-token", int(time.time()) + 3600)

    def test_returns_503_and_spawns_sidecar_on_401_with_expired_token(self, _setup_token):
        """Upstream 401 + expired token → 503 + sidecar spawn."""
        expired_token = _make_jwt(exp_offset_seconds=-3600)
        _setup_token(expired_token, int(time.time()) - 3600)

        async def mock_create(**kwargs):
            raise openai.APIStatusError(
                message="Unauthorized",
                response=MagicMock(status_code=401),
                body=None,
            )

        with patch("src.proxy.openai.routes.request_refresh") as mock_trigger:
            with TestClient(_app_with(mock_create)) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                )

            assert resp.status_code == 503
            error = resp.json()["error"]
            assert "expired" in error["message"].lower() or "refresh" in error["message"].lower()
            mock_trigger.assert_called_once()

    def test_returns_401_without_spawning_when_token_not_expired(self, _setup_token):
        """Upstream 401 but token fresh → pass through 401, no refresh."""
        fresh_token = _make_jwt(exp_offset_seconds=3600)
        _setup_token(fresh_token, int(time.time()) + 3600)

        async def mock_create(**kwargs):
            raise openai.APIStatusError(
                message="Unauthorized",
                response=MagicMock(status_code=401),
                body=None,
            )

        with patch("src.proxy.openai.routes.request_refresh") as mock_trigger:
            with TestClient(_app_with(mock_create)) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": True},
                )

            assert resp.status_code == 401
            mock_trigger.assert_not_called()

    def test_non_streaming_returns_503_on_401_expired_token(self, _setup_token):
        """Non-streaming endpoint also triggers refresh on 401+expired."""
        expired_token = _make_jwt(exp_offset_seconds=-3600)
        _setup_token(expired_token, int(time.time()) - 3600)

        async def mock_create(**kwargs):
            raise openai.APIStatusError(
                message="Unauthorized",
                response=MagicMock(status_code=401),
                body=None,
            )

        with patch("src.proxy.openai.routes.request_refresh") as mock_trigger:
            with TestClient(_app_with(mock_create)) as client:
                resp = client.post(
                    "/v1/chat/completions",
                    json={"model": "test", "messages": [{"role": "user", "content": "hi"}], "stream": False},
                )

            assert resp.status_code == 503
            mock_trigger.assert_called_once()


class TestRefreshLock:
    """Tests for the refresh lock file mechanism."""

    def test_lock_file_path_exists(self):
        from src.auth import REFRESH_LOCK_PATH
        assert REFRESH_LOCK_PATH is not None

    def test_trigger_refresh_runs_sidecar(self, tmp_path: Path):
        lock_path = tmp_path / "test-lock.lock"
        sidecar_called = False

        def fake_popen(*args, **kwargs):
            nonlocal sidecar_called
            sidecar_called = True
            return MagicMock()

        with patch("src.auth.REFRESH_LOCK_PATH", lock_path):
            with patch("subprocess.Popen", fake_popen):
                from src.auth import request_refresh
                request_refresh()
                assert sidecar_called
                assert lock_path.exists() is False


class TestShouldRefreshToken:
    """A 401 must only trigger refresh on positive evidence the token is at fault."""

    def test_true_when_token_expired(self):
        assert should_refresh(_make_jwt(exp_offset_seconds=-3600), None) is True

    def test_false_when_token_fresh_and_no_body(self):
        assert should_refresh(_make_jwt(exp_offset_seconds=3600), None) is False

    def test_true_on_invalid_issuer_even_when_token_looks_fresh(self):
        fresh = _make_jwt(exp_offset_seconds=3600)
        assert should_refresh(fresh, {"code": "invalid_issuer"}) is True

    def test_true_on_nested_error_code(self):
        fresh = _make_jwt(exp_offset_seconds=3600)
        assert should_refresh(fresh, {"error": {"code": "invalid_token"}}) is True

    def test_false_on_unrelated_401_code(self):
        fresh = _make_jwt(exp_offset_seconds=3600)
        assert should_refresh(fresh, {"code": "model_access_denied"}) is False

    def test_false_on_non_dict_body(self):
        fresh = _make_jwt(exp_offset_seconds=3600)
        assert should_refresh(fresh, "upstream said no") is False


class TestSingleFlightOwnership:
    def test_spawn_does_not_hold_the_lock_the_sidecar_needs(self, tmp_path: Path):
        import fcntl

        lock_path = tmp_path / "refresh.lock"
        held_by_parent: list[bool] = []

        def fake_popen(*args, **kwargs):
            probe = os.open(str(lock_path), os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held_by_parent.append(False)
                fcntl.flock(probe, fcntl.LOCK_UN)
            except BlockingIOError:
                held_by_parent.append(True)
            finally:
                os.close(probe)
            return MagicMock()

        with patch("src.auth.REFRESH_LOCK_PATH", lock_path):
            with patch("subprocess.Popen", fake_popen):
                from src.auth import request_refresh
                assert request_refresh() is True

        assert held_by_parent == [False]

    def test_lock_path_is_not_unlinked_by_the_proxy(self, tmp_path: Path):
        lock_path = tmp_path / "refresh.lock"
        lock_path.touch()

        with patch("src.auth.REFRESH_LOCK_PATH", lock_path):
            with patch("subprocess.Popen", lambda *a, **k: MagicMock()):
                from src.auth import request_refresh
                request_refresh()

        assert lock_path.exists()


class TestAtomicTokenWrite:
    def test_write_is_atomic_and_mode_0600(self, tmp_path: Path):
        import stat

        from src.auth import write_token_file

        target = tmp_path / "nested" / "token.json"
        write_token_file("tok-abc", 1234567890, path=target)

        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        payload = json.loads(target.read_text())
        assert payload["token"] == "tok-abc"
        assert payload["expires_at"] == 1234567890
        assert "retrieved_at" in payload

    def test_no_temp_files_left_behind(self, tmp_path: Path):
        from src.auth import write_token_file

        target = tmp_path / "token.json"
        write_token_file("tok", None, path=target)
        assert [p.name for p in tmp_path.iterdir()] == ["token.json"]

    def test_reader_never_sees_partial_json(self, tmp_path: Path):
        from src.auth import write_token_file

        target = tmp_path / "token.json"
        write_token_file("first-token", 1, path=target)
        original_inode = target.stat().st_ino

        write_token_file("second-token", 2, path=target)

        assert json.loads(target.read_text())["token"] == "second-token"
        assert target.stat().st_ino != original_inode


class TestSidecarReaping:
    def test_spawned_sidecar_is_waited_on(self, tmp_path: Path):
        waited: list[bool] = []

        class FakeProc:
            def wait(self):
                waited.append(True)
                return 0

        with patch("src.auth.REFRESH_LOCK_PATH", tmp_path / "l.lock"):
            with patch("subprocess.Popen", lambda *a, **k: FakeProc()):
                from src.auth import request_refresh
                assert request_refresh() is True

        for _ in range(200):
            if waited:
                break
            time.sleep(0.01)
        assert waited == [True]
