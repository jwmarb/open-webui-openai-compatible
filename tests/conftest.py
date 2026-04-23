import os
import pytest

# Set env vars at module level BEFORE any src imports.
# This prevents pydantic-settings from raising ValidationError during test collection
# when src/settings.py executes `settings = Settings()` at import time.
os.environ.setdefault("OPEN_WEBUI_URL", "https://test.example.com")
os.environ.setdefault("USER_TOKEN", "test-token-123")
os.environ.setdefault("PORT", "8000")


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Ensure test environment variables are always set for every test."""
    monkeypatch.setenv("OPEN_WEBUI_URL", "https://test.example.com")
    monkeypatch.setenv("USER_TOKEN", "test-token-123")
    monkeypatch.setenv("PORT", "8000")
