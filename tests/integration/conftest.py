import time

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.settings import settings
from tests.conftest import TEST_DEFAULT_TOKEN, TEST_DEFAULT_URL

MODEL_LIST_RETRIES = 5
MODEL_LIST_BACKOFF = 3


def _is_real_instance() -> bool:
    return (
        settings.open_webui_url != TEST_DEFAULT_URL
        and settings.user_token != TEST_DEFAULT_TOKEN
    )


skip_without_real_instance = pytest.mark.skipif(
    not _is_real_instance(),
    reason="Integration tests require real OPEN_WEBUI_URL and USER_TOKEN env vars",
)


def fetch_models_with_retry(client_or_openai):
    """Fetch model list, retrying on transient empty responses from upstream.

    Accepts either a TestClient (returns raw JSON data list) or an OpenAI client
    (returns SyncPage[Model]).
    """
    from openai import OpenAI

    for attempt in range(MODEL_LIST_RETRIES):
        if isinstance(client_or_openai, OpenAI):
            result = client_or_openai.models.list()
            if len(result.data) > 0:
                return result
        else:
            resp = client_or_openai.get("/v1/models")
            body = resp.json()
            if len(body.get("data", [])) > 0:
                return body
        if attempt < MODEL_LIST_RETRIES - 1:
            time.sleep(MODEL_LIST_BACKOFF)

    pytest.fail("No models available from upstream after retries")


@pytest.fixture
def client():
    with TestClient(app) as tc:
        yield tc
