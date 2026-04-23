import os

import pytest

# Shared test defaults — integration conftest references these to detect
# whether real credentials were provided.
TEST_DEFAULT_URL = "https://test.example.com"
TEST_DEFAULT_TOKEN = "test-token-123"
TEST_DEFAULT_PORT = "8000"

# Set env vars at module level BEFORE any src imports.
# This prevents pydantic-settings from raising ValidationError during test collection
# when src/settings.py executes `settings = Settings()` at import time.
os.environ.setdefault("OPEN_WEBUI_URL", TEST_DEFAULT_URL)
os.environ.setdefault("USER_TOKEN", TEST_DEFAULT_TOKEN)
os.environ.setdefault("PORT", TEST_DEFAULT_PORT)


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Ensure test environment variables are always set for every test."""
    monkeypatch.setenv("OPEN_WEBUI_URL", TEST_DEFAULT_URL)
    monkeypatch.setenv("USER_TOKEN", TEST_DEFAULT_TOKEN)
    monkeypatch.setenv("PORT", TEST_DEFAULT_PORT)
