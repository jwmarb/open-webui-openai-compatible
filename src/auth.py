"""Token provider — reads JWT from file or environment, checks expiry."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import jwt

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE = Path.home() / ".config" / "open-webui-proxy" / "token.json"


def _get_token_file_path() -> Path:
    env_path = os.environ.get("TOKEN_FILE")
    if env_path:
        return Path(env_path)
    return DEFAULT_TOKEN_FILE


def get_token_expiry(token: str | None) -> int | None:
    if not token:
        return None
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        return payload.get("exp")
    except Exception:
        return None


def is_token_expired_or_invalid(token: str | None) -> bool:
    """True only when the token is a decodable JWT whose exp has passed.

    A malformed token returns False: re-running the browser login cannot repair a
    garbled token, so it must not be treated as grounds for a refresh.
    """
    if not token:
        return False
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
    except Exception:
        return False
    exp = payload.get("exp")
    if exp is None:
        return False
    return time.time() > exp


def is_token_expired(token: str | None) -> bool:
    if not token:
        return False
    try:
        payload = jwt.decode(token, options={"verify_signature": False})
        exp = payload.get("exp")
        if exp is None:
            return False
        return time.time() > exp
    except Exception:
        return False


def _read_token_from_file(path: Path) -> str | None:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        token = data.get("token")
        if isinstance(token, str) and token:
            return token
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.debug("Failed to read token from %s: %s", path, exc)
    return None


def _read_token_from_env() -> str | None:
    token = os.environ.get("USER_TOKEN")
    return token if token else None


def get_current_token() -> str:
    token_file = _get_token_file_path()

    if token_file.exists():
        token = _read_token_from_file(token_file)
        if token:
            return token

    token = _read_token_from_env()
    if token:
        return token

    raise RuntimeError(
        "No authentication token available. "
        "Set USER_TOKEN environment variable or provide a token file at "
        f"{token_file} via the TOKEN_FILE environment variable."
    )
