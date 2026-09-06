"""Tests for src/auth.py — token provider, JWT expiry checking, fallback logic."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import jwt
import pytest

from src.auth import get_token_expiry, is_token_expired_or_invalid

DEFAULT_TOKEN_FILE = Path.home() / ".config" / "open-webui-proxy" / "token.json"


def _make_jwt(exp_offset_seconds: int, extra: dict[str, Any] | None = None) -> str:
    """Create a test JWT with a known exp claim."""
    now = int(time.time())
    payload = {"id": "test-user", "iat": now, "exp": now + exp_offset_seconds}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, "test-secret", algorithm="HS256")


class TestGetTokenExpiry:
    """Tests for get_token_expiry — reads exp claim from JWT."""

    def test_returns_expiry_for_valid_token(self):
        now = int(time.time())
        token = _make_jwt(exp_offset_seconds=3600)
        exp = get_token_expiry(token)
        assert exp is not None
        assert now < exp <= now + 3600

    def test_returns_none_for_token_without_exp(self):
        payload = {"id": "test-user", "iat": int(time.time())}
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        assert get_token_expiry(token) is None

    def test_returns_none_for_invalid_token(self):
        assert get_token_expiry("not-a-jwt") is None
        assert get_token_expiry("") is None

    def test_returns_none_for_none_token(self):
        assert get_token_expiry(None) is None  # type: ignore[arg-type]


class TestIsTokenExpired:
    """Tests for is_token_expired_or_invalid — checks if JWT exp claim has passed."""

    def test_returns_false_for_fresh_token(self):
        token = _make_jwt(exp_offset_seconds=3600)
        assert is_token_expired_or_invalid(token) is False

    def test_returns_true_for_expired_token(self):
        token = _make_jwt(exp_offset_seconds=-3600)
        assert is_token_expired_or_invalid(token) is True

    def test_returns_false_for_token_without_exp(self):
        payload = {"id": "test-user", "iat": int(time.time())}
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        assert is_token_expired_or_invalid(token) is False

    def test_returns_false_for_invalid_token(self):
        assert is_token_expired_or_invalid("not-a-jwt") is False

    def test_returns_false_for_none_token(self):
        assert is_token_expired_or_invalid(None) is False  # type: ignore[arg-type]


class TestIsTokenExpiredOrInvalid:
    """Only a decodable JWT past its exp counts — malformed tokens are not refreshable."""

    def test_returns_true_for_expired_token(self):
        assert is_token_expired_or_invalid(_make_jwt(exp_offset_seconds=-3600)) is True

    def test_returns_false_for_fresh_token(self):
        assert is_token_expired_or_invalid(_make_jwt(exp_offset_seconds=3600)) is False

    def test_returns_false_for_malformed_token(self):
        assert is_token_expired_or_invalid("not-a-jwt") is False

    def test_returns_false_for_token_without_exp(self):
        payload = {"id": "test-user", "iat": int(time.time())}
        assert is_token_expired_or_invalid(jwt.encode(payload, "s", algorithm="HS256")) is False

    def test_returns_false_for_none_token(self):
        assert is_token_expired_or_invalid(None) is False


class TestGetCurrentToken:
    """Tests for get_current_token — file → env fallback chain."""

    def setup_method(self):
        self.original_token_file = os.environ.pop("TOKEN_FILE", None)

    def teardown_method(self):
        if self.original_token_file is not None:
            os.environ["TOKEN_FILE"] = self.original_token_file
        elif "TOKEN_FILE" in os.environ:
            del os.environ["TOKEN_FILE"]

    def test_returns_token_from_file_when_exists(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text(
            json.dumps({"token": "file-token-123", "expires_at": 9999999999})
        )
        monkeypatch.setenv("TOKEN_FILE", str(token_file))
        monkeypatch.setenv("USER_TOKEN", "env-token-456")

        import importlib

        import src.auth

        importlib.reload(src.auth)
        assert src.auth.get_current_token() == "file-token-123"

    def test_falls_back_to_env_when_file_missing(self, monkeypatch):
        monkeypatch.setenv("TOKEN_FILE", "/nonexistent/path/token.json")
        monkeypatch.setenv("USER_TOKEN", "env-token-456")

        import importlib

        import src.auth

        importlib.reload(src.auth)
        assert src.auth.get_current_token() == "env-token-456"

    def test_falls_back_to_env_when_file_invalid_json(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text("not json {")
        monkeypatch.setenv("TOKEN_FILE", str(token_file))
        monkeypatch.setenv("USER_TOKEN", "env-token-456")

        import importlib

        import src.auth

        importlib.reload(src.auth)
        assert src.auth.get_current_token() == "env-token-456"

    def test_falls_back_to_env_when_file_missing_token_field(self, tmp_path, monkeypatch):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({"expires_at": 9999999999}))
        monkeypatch.setenv("TOKEN_FILE", str(token_file))
        monkeypatch.setenv("USER_TOKEN", "env-token-456")

        import importlib

        import src.auth

        importlib.reload(src.auth)
        assert src.auth.get_current_token() == "env-token-456"

    def test_raises_when_no_file_and_no_env(self, monkeypatch):
        monkeypatch.setenv("TOKEN_FILE", "/nonexistent/token.json")
        monkeypatch.delenv("USER_TOKEN", raising=False)

        import importlib

        import src.auth

        importlib.reload(src.auth)
        with pytest.raises(RuntimeError, match="No authentication token"):
            src.auth.get_current_token()

    def test_uses_default_path_when_token_file_env_not_set(self, monkeypatch, tmp_path):
        mock_home = tmp_path / "home"
        mock_home.mkdir()
        config_dir = mock_home / ".config" / "open-webui-proxy"
        config_dir.mkdir(parents=True)
        token_file = config_dir / "token.json"
        token_file.write_text(
            json.dumps({"token": "default-path-token", "expires_at": 9999999999})
        )

        monkeypatch.delenv("TOKEN_FILE", raising=False)
        monkeypatch.setenv("USER_TOKEN", "env-token-should-be-ignored")
        monkeypatch.setattr(Path, "home", lambda: mock_home)

        import importlib

        import src.auth

        importlib.reload(src.auth)
        assert src.auth.get_current_token() == "default-path-token"
