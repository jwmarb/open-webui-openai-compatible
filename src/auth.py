"""Token store and refresh protocol.

The only interface callers need for credentials:

- `get_current_token()` — the token on disk or in the environment.
- `request_refresh()` — ask for a single-flight renewal; returns whether a
  sidecar was spawned.
- `should_refresh(token, body)` — is an upstream rejection token-related?

Single-flight ownership belongs to the spawned sidecar, never to the caller:
the sidecar acquires the lock for the entire refresh and losing sidecars exit
as no-ops. A caller that held the lock across the spawn would block the very
child it just started.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Final

import jwt

logger = logging.getLogger(__name__)

DEFAULT_TOKEN_FILE = Path.home() / ".config" / "open-webui-proxy" / "token.json"
REFRESH_LOCK_PATH: Final[Path] = Path("/tmp/openwebui-proxy-refresh.lock")
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
REFRESH_SCRIPT: Final[Path] = PROJECT_ROOT / "playwright_login.py"
SIDECAR_LOG_PATH: Final[Path] = Path("/tmp/sidecar.log")

TOKEN_KEY: Final[str] = "token"
EXPIRES_AT_KEY: Final[str] = "expires_at"
RETRIEVED_AT_KEY: Final[str] = "retrieved_at"
TOKEN_FILE_MODE: Final[int] = 0o600

_TOKEN_REJECTION_CODES: Final[frozenset[str]] = frozenset({
    "invalid_issuer",
    "invalid_token",
    "token_expired",
})


def get_token_file_path() -> Path:
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

    A malformed token returns False. The browser login would in fact overwrite
    it, but treating local corruption as grounds for renewal turns every
    request into a refresh storm, so it must not trigger one.
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


def write_token_file(token: str, expires_at: int | None, path: Path | None = None) -> Path:
    """Atomically replace the token file, mode 0600 from creation."""
    token_path = path or get_token_file_path()
    token_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        TOKEN_KEY: token,
        EXPIRES_AT_KEY: expires_at,
        RETRIEVED_AT_KEY: int(time.time()),
    }

    fd, tmp_name = tempfile.mkstemp(dir=str(token_path.parent), prefix=".token-", suffix=".tmp")
    try:
        os.fchmod(fd, TOKEN_FILE_MODE)
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, token_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return token_path


def _read_token_from_file(path: Path) -> str | None:
    try:
        data: dict[str, Any] = json.loads(path.read_text())
        token = data.get(TOKEN_KEY)
        if isinstance(token, str) and token:
            return token
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        logger.debug("Failed to read token from %s: %s", path, exc)
    return None


def _read_token_from_env() -> str | None:
    token = os.environ.get("USER_TOKEN")
    return token if token else None


def get_current_token() -> str:
    token_file = get_token_file_path()

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


def extract_error_code(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    code = body.get("code")
    if isinstance(code, str):
        return code
    error = body.get("error")
    if isinstance(error, dict):
        nested = error.get("code")
        if isinstance(nested, str):
            return nested
    return None


def should_refresh(token: str | None, body: Any) -> bool:
    """A 401 alone is ambiguous; require positive evidence of a token fault."""
    if is_token_expired_or_invalid(token):
        return True
    return extract_error_code(body) in _TOKEN_REJECTION_CODES


def _reap(process: subprocess.Popen[bytes]) -> None:
    """Wait on a detached sidecar so it does not linger as a zombie.

    uvicorn runs as PID 1 in the container image and does not reap. A detached
    Popen that is never waited on therefore accumulates zombies. Waiting on a
    daemon thread keeps the event loop unblocked.
    """
    thread = threading.Thread(target=process.wait, daemon=True, name="sidecar-reaper")
    thread.start()


def request_refresh() -> bool:
    """Spawn the refresh sidecar, which owns the single-flight lock itself.

    Returns True when a sidecar process was started. Concurrent callers may
    each spawn one; every loser exits immediately as a no-op.
    """
    if not REFRESH_SCRIPT.exists():
        logger.error("Refresh sidecar not found at %s", REFRESH_SCRIPT)
        return False

    try:
        log_handle = open(SIDECAR_LOG_PATH, "a")
    except OSError as exc:
        logger.error("Cannot open sidecar log %s: %s", SIDECAR_LOG_PATH, exc)
        return False

    try:
        process = subprocess.Popen(
            [sys.executable, str(REFRESH_SCRIPT)],
            stdout=log_handle,
            stderr=log_handle,
            start_new_session=True,
        )
    except Exception as exc:
        logger.error("Failed to spawn refresh sidecar: %s", exc)
        return False
    finally:
        log_handle.close()

    _reap(process)

    logger.info("Token refresh sidecar spawned")
    return True
