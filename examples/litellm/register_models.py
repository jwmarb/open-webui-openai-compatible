"""LiteLLM gateway startup hook: teach the model cost map about gateway model IDs.

LiteLLM's anthropic provider validates ``thinking`` / ``reasoning_effort``
against its model capability map. The gateway's Bedrock-style IDs
(``bedrock-claude-5-opus`` ...) do not match the ``claude-<family>-<version>``
patterns LiteLLM's fallback rules expect, so the provider rejects both
parameters with ``UnsupportedParamsError`` before the request ever reaches
this proxy.

Registering the IDs with explicit capability flags fixes that. The
registration MUST happen at module import time: LiteLLM resolves the dotted
``success_callback`` path via ``getattr`` at startup, which triggers the
module import and the ``register_model`` call below. The class defined here
is a placeholder that satisfies the callback mechanism; it is not a logger.

Wiring (LiteLLM gateway config YAML)::

    model_list:
      - model_name: bedrock-claude-5-opus
        litellm_params:
          model: anthropic/bedrock-claude-5-opus
          api_base: http://<proxy-host>:8000   # no /v1 — LiteLLM appends /v1/messages
          api_key: dummy                       # the proxy authenticates upstream itself
    litellm_settings:
      success_callback: ["register_models.RegisterGatewayModels"]

Run the gateway with this file's directory on PYTHONPATH.

Standalone by design: imports only ``litellm``, nothing from ``src/``
(same independence principle as ``tui.py``).
"""

import litellm

# One entry per Claude model the gateway serves. 4.6 families keep the legacy
# ``thinking.type="enabled" + budget_tokens`` shape too; 5.x is adaptive-only.
# ``fable`` and ``mythos`` are omitted: they are not served by this gateway.
_GATEWAY_MODELS: dict[str, dict[str, object]] = {
    "bedrock-claude-5-opus": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_output_config": True,
        "supports_xhigh_reasoning_effort": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
    "bedrock-claude-5-sonnet": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_output_config": True,
        "supports_xhigh_reasoning_effort": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
    "bedrock-claude-5-5-opus": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_output_config": True,
        "supports_xhigh_reasoning_effort": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
    "bedrock-claude-5-5-sonnet": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_output_config": True,
        "supports_xhigh_reasoning_effort": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
    "bedrock-claude-4-6-opus": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_legacy_thinking": True,
        "supports_output_config": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
    "bedrock-claude-4-6-sonnet": {
        "litellm_provider": "anthropic",
        "mode": "chat",
        "supports_adaptive_thinking": True,
        "supports_legacy_thinking": True,
        "supports_output_config": True,
        "max_output_tokens": 64000,
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
    },
}

# Register at import time: the proxy imports this module at startup, so the
# cost map is populated before any request is handled.
litellm.register_model(_GATEWAY_MODELS)


class RegisterGatewayModels:
    """Placeholder success callback; the module import is the real side effect.

    LiteLLM's callback resolution ``getattr``s this class from the module and
    appends it to ``litellm.success_callback``. On each successful request the
    logging pipeline calls it with ``(kwargs, response, start_time, end_time)``;
    we accept and discard those args.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass
