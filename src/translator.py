# Backward-compatibility re-export — canonical location is src.proxy.openai.translator
from .errors import create_openai_error  # noqa: F401
from .proxy.openai.translator import (  # noqa: F401
    ADAPTIVE_THINKING_CONFIG,
    EXTENDED_THINKING_CONFIG,
    EXTENDED_THINKING_CONFIG_SMALL,
    MIN_MAX_TOKENS_EXTENDED,
    MIN_MAX_TOKENS_EXTENDED_SMALL,
    THINKING_SUFFIX_ADAPTIVE,
    THINKING_SUFFIX_EXTENDED,
    apply_thinking_params,
    generate_thinking_variants,
    resolve_thinking_model,
    rewrite_chat_body,
    sanitize_chat_body,
    translate_models_response,
)
