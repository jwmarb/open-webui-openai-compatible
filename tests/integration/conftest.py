import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.settings import settings


def _is_real_instance() -> bool:
    """Check if settings point to a real Open WebUI instance (not test defaults)."""
    return (
        settings.open_webui_url != "https://test.example.com"
        and settings.user_token != "test-token-123"
    )


skip_without_real_instance = pytest.mark.skipif(
    not _is_real_instance(),
    reason="Integration tests require real OPEN_WEBUI_URL and USER_TOKEN env vars",
)


@pytest.fixture
def client():
    """TestClient that talks to the real FastAPI app (which talks to real upstream)."""
    with TestClient(app) as tc:
        yield tc
