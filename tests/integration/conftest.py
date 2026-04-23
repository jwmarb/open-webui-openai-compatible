import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.settings import settings
from tests.conftest import TEST_DEFAULT_TOKEN, TEST_DEFAULT_URL


def _is_real_instance() -> bool:
    return (
        settings.open_webui_url != TEST_DEFAULT_URL
        and settings.user_token != TEST_DEFAULT_TOKEN
    )


skip_without_real_instance = pytest.mark.skipif(
    not _is_real_instance(),
    reason="Integration tests require real OPEN_WEBUI_URL and USER_TOKEN env vars",
)


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc
