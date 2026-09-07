# Backward-compatibility re-export — canonical location is src.proxy.openai.translator
from .proxy.openai.errors import create_openai_error  # noqa: F401
from .proxy.openai.translator import (  # noqa: F401
    ADAPTIVE_THINKING_CONFIG,
    MIN_MAX_TOKENS_ADAPTIVE,
    THINKING_SUFFIX_ADAPTIVE,
    apply_thinking_params,
    generate_thinking_variants,
    resolve_thinking_model,
    rewrite_chat_body,
    sanitize_chat_body,
    translate_models_response,
)
