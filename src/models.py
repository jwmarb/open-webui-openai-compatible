# Backward-compatibility re-export — canonical location is src.proxy.openai.models
from .proxy.openai.models import (  # noqa: F401
    OpenAIErrorDetail,
    OpenAIErrorResponse,
    OpenAIModel,
    OpenAIModelList,
    ThinkingConfig,
)
