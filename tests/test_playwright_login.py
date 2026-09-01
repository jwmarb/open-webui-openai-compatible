"""Tests for playwright-login.py sidecar — browser launch, token extraction."""

from __future__ import annotations

import json
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import pytest

# Import functions from the sidecar script
import playwright_login as sidecar


@pytest.fixture
def token_dir(tmp_path: Path) -> Path:
    d = tmp_path / "token-dir"
    d.mkdir()
    return d


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    d = tmp_path / "browser-profile"
    d.mkdir()
    return d


class MockHandler(SimpleHTTPRequestHandler):
    """Serves a page that sets the `token` cookie, mirroring Open WebUI's real behavior."""

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Set-Cookie", "token=test-jwt-token-from-browser; Path=/")
            self.end_headers()
            self.wfile.write(b"<!DOCTYPE html><html><body>Logged in</body></html>")
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class MockShibbolethHandler(SimpleHTTPRequestHandler):
    """Simulates the UofA Shibboleth login form, then sets the session cookie on POST."""

    def do_GET(self):
        html = """
        <!DOCTYPE html>
        <html><body>
        <form method="POST" action="/login">
            <input type="text" id="username" name="j_username">
            <input type="password" id="password" name="j_password">
            <button type="submit">Login</button>
        </form>
        </body></html>
        """
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Set-Cookie", "token=test-jwt-after-login; Path=/")
        self.end_headers()
        self.wfile.write(b"<!DOCTYPE html><html><body>Logged in after Shibboleth</body></html>")

    def log_message(self, format, *args):
        pass


@pytest.fixture
def mock_server():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def mock_shibboleth_server():
    server = HTTPServer(("127.0.0.1", 0), MockShibbolethHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestSidecarTokenExtraction:
    """Tests for the sidecar's core token extraction logic."""

    def test_extract_token_from_localstorage(self, mock_server: str, token_dir: Path, profile_dir: Path):
        """Sidecar extracts JWT from localStorage and writes token file."""
        with (
            patch.dict(os.environ, {
                "OPEN_WEBUI_URL": mock_server,
                "TOKEN_FILE": str(token_dir / "token.json"),
                "BROWSER_PROFILE_DIR": str(profile_dir),
            }),
        ):
            success = sidecar.run()
            assert success is True
            token_file = token_dir / "token.json"
            assert token_file.exists()
            data = json.loads(token_file.read_text())
            assert data["token"] == "test-jwt-token-from-browser"
            assert "expires_at" in data
            assert "retrieved_at" in data

    def test_writes_token_file_with_correct_permissions(self, mock_server: str, token_dir: Path, profile_dir: Path):
        """Token file is written with owner-only permissions (0600)."""
        with patch.dict(os.environ, {
            "OPEN_WEBUI_URL": mock_server,
            "TOKEN_FILE": str(token_dir / "token.json"),
            "BROWSER_PROFILE_DIR": str(profile_dir),
        }):
            sidecar.run()
            token_file = token_dir / "token.json"
            mode = oct(token_file.stat().st_mode)[-3:]
            assert mode == "600"

    def test_fails_when_no_token_found(self, token_dir: Path, profile_dir: Path):
        """Sidecar fails when page doesn't set localStorage.token."""
        server = HTTPServer(("127.0.0.1", 0), SimpleHTTPRequestHandler)
        port = server.server_address[1]
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            with patch.dict(os.environ, {
                "OPEN_WEBUI_URL": f"http://127.0.0.1:{port}",
                "TOKEN_FILE": str(token_dir / "token.json"),
                "BROWSER_PROFILE_DIR": str(profile_dir),
            }):
                success = sidecar.run()
                assert success is False
        finally:
            server.shutdown()


class TestSidecarUofaLogin:
    """Tests for UofA Shibboleth credential filling."""

    def test_fills_credentials_and_extracts_token(
        self, mock_shibboleth_server: str, token_dir: Path, profile_dir: Path
    ):
        """Sidecar fills UofA login form and extracts token after auth."""
        with patch.dict(os.environ, {
            "OPEN_WEBUI_URL": mock_shibboleth_server,
            "UA_NETID": "testnetid",
            "UA_NETID_PASSWORD": "testpass",
            "TOKEN_FILE": str(token_dir / "token.json"),
            "BROWSER_PROFILE_DIR": str(profile_dir),
        }):
            success = sidecar.run()
            assert success is True
            token_file = token_dir / "token.json"
            assert token_file.exists()
            data = json.loads(token_file.read_text())
            assert data["token"] == "test-jwt-after-login"
