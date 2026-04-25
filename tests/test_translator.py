from src.models import OpenAIModel, ThinkingConfig
from src.translator import (
    ADAPTIVE_THINKING_CONFIG,
    EXTENDED_THINKING_CONFIG,
    apply_thinking_params,
    create_openai_error,
    generate_thinking_variants,
    resolve_thinking_model,
    sanitize_chat_body,
    translate_models_response,
)


class TestTranslateModelsResponse:

    def test_translate_empty_response(self):
        raw = {"data": []}
        result = translate_models_response(raw)
        assert result == {"object": "list", "data": []}

    def test_translate_empty_dict(self):
        raw = {}
        result = translate_models_response(raw)
        assert result == {"object": "list", "data": []}

    def test_translate_single_model(self):
        raw = {
            "data": [
                {
                    "id": "llama-3.1-8b",
                    "name": "Llama 3.1 8B",
                    "owned_by": "Meta",
                    "created": 1700000000,
                    "extra_field": "ignored",
                }
            ]
        }
        result = translate_models_response(raw)
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        model = result["data"][0]
        assert model == {
            "id": "llama-3.1-8b",
            "object": "model",
            "created": 1700000000,
            "owned_by": "Meta",
        }
        assert "name" not in model
        assert "extra_field" not in model

    def test_translate_multiple_models(self):
        raw = {
            "data": [
                {"id": "gpt-4", "owned_by": "OpenAI", "created": 1690000000},
                {"id": "claude-3", "owned_by": "Anthropic"},
                {"id": "mistral-7b", "created": None, "owned_by": ""},
            ]
        }
        result = translate_models_response(raw)
        ids = [m["id"] for m in result["data"]]
        assert "gpt-4" in ids
        assert "claude-3" in ids
        assert "claude-3:extended" in ids
        assert "claude-3:adaptive" in ids
        assert "mistral-7b" in ids
        assert result["data"][0]["id"] == "gpt-4"

    def test_translate_missing_fields(self):
        raw = {"data": [{}]}
        result = translate_models_response(raw)
        assert len(result["data"]) == 1
        assert result["data"][0] == {
            "id": "",
            "object": "model",
            "created": 0,
            "owned_by": "",
        }


class TestCreateOpenAIError:

    def test_create_openai_error_default(self):
        result = create_openai_error("Something went wrong")
        assert result == {
            "error": {
                "message": "Something went wrong",
                "type": "invalid_request_error",
                "code": None,
            }
        }

    def test_create_openai_error_custom_type_and_code(self):
        result = create_openai_error(
            "Upstream request failed",
            error_type="api_error",
            code=502,
        )
        assert result == {
            "error": {
                "message": "Upstream request failed",
                "type": "api_error",
                "code": 502,
            }
        }


class TestSanitizeChatBody:

    def test_keeps_standard_openai_fields(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0.7,
            "tools": [],
            "tool_choice": "auto",
        }
        assert sanitize_chat_body(body) == body

    def test_strips_litellm_fields(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {},
            "extra_headers": {},
            "api_base": "http://example.com",
            "api_key": "sk-test",
            "custom_llm_provider": "openai",
            "litellm_call_id": "abc-123",
            "litellm_logging_obj": {},
        }
        result = sanitize_chat_body(body)
        assert result == {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_empty_body(self):
        assert sanitize_chat_body({}) == {}

    def test_preserves_all_openai_params(self):
        body = {
            "model": "gpt-4",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.5,
            "top_p": 0.9,
            "n": 1,
            "stop": ["\n"],
            "max_tokens": 100,
            "max_completion_tokens": 100,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "logit_bias": {},
            "logprobs": True,
            "top_logprobs": 5,
            "user": "user-1",
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "response_format": {"type": "json_object"},
            "seed": 42,
            "service_tier": "default",
            "metadata": {},
            "store": True,
            "reasoning_effort": "medium",
            "functions": [],
            "function_call": "auto",
        }
        assert sanitize_chat_body(body) == body

    def test_strips_all_non_openai_fields(self):
        body = {"extra_body": {}, "api_base": "http://x", "custom_llm_provider": "openai", "litellm_call_id": "abc"}
        assert sanitize_chat_body(body) == {}


class TestGenerateThinkingVariants:

    def test_non_claude_model_no_variants(self):
        model = OpenAIModel(id="gpt-4", owned_by="OpenAI")
        assert generate_thinking_variants(model) == []

    def test_claude_opus_gets_extended_and_adaptive(self):
        model = OpenAIModel(id="bedrock-claude-4-6-opus")
        variants = generate_thinking_variants(model)
        ids = [v.id for v in variants]
        assert ids == ["bedrock-claude-4-6-opus:extended", "bedrock-claude-4-6-opus:adaptive"]

    def test_claude_sonnet_gets_extended_and_adaptive(self):
        model = OpenAIModel(id="bedrock-claude-4-5-sonnet")
        variants = generate_thinking_variants(model)
        ids = [v.id for v in variants]
        assert ids == ["bedrock-claude-4-5-sonnet:extended", "bedrock-claude-4-5-sonnet:adaptive"]

    def test_claude_haiku_gets_extended_only(self):
        model = OpenAIModel(id="bedrock-claude-4-5-haiku")
        variants = generate_thinking_variants(model)
        ids = [v.id for v in variants]
        assert ids == ["bedrock-claude-4-5-haiku:extended"]

    def test_variant_preserves_created_and_owned_by(self):
        model = OpenAIModel(id="bedrock-claude-4-6-opus", created=1700000000, owned_by="Anthropic")
        variants = generate_thinking_variants(model)
        for v in variants:
            assert v.created == 1700000000
            assert v.owned_by == "Anthropic"
            assert v.object == "model"

    def test_models_response_includes_variants(self):
        raw = {"data": [
            {"id": "bedrock-claude-4-5-haiku", "owned_by": ""},
            {"id": "gpt-4", "owned_by": "OpenAI"},
        ]}
        result = translate_models_response(raw)
        ids = [m["id"] for m in result["data"]]
        assert ids == [
            "bedrock-claude-4-5-haiku",
            "bedrock-claude-4-5-haiku:extended",
            "gpt-4",
        ]


class TestResolveThinkingModel:

    def test_plain_model_no_thinking(self):
        base, config = resolve_thinking_model("bedrock-claude-4-6-opus")
        assert base == "bedrock-claude-4-6-opus"
        assert config is None

    def test_extended_suffix_stripped(self):
        base, config = resolve_thinking_model("bedrock-claude-4-6-opus:extended")
        assert base == "bedrock-claude-4-6-opus"
        assert config is not None
        assert config.type == "enabled"
        assert config.budget_tokens == 32000

    def test_adaptive_suffix_stripped(self):
        base, config = resolve_thinking_model("bedrock-claude-4-5-sonnet:adaptive")
        assert base == "bedrock-claude-4-5-sonnet"
        assert config == ThinkingConfig(type="adaptive")

    def test_haiku_extended_gets_smaller_budget(self):
        base, config = resolve_thinking_model("bedrock-claude-4-5-haiku:extended")
        assert base == "bedrock-claude-4-5-haiku"
        assert config is not None
        assert config.budget_tokens == 16000

    def test_non_claude_with_colon_unchanged(self):
        base, config = resolve_thinking_model("some-model:v2")
        assert base == "some-model:v2"
        assert config is None

    def test_non_claude_extended_suffix_still_resolves(self):
        base, config = resolve_thinking_model("gpt-4:extended")
        assert base == "gpt-4"
        assert config == EXTENDED_THINKING_CONFIG

    def test_non_claude_adaptive_suffix_still_resolves(self):
        base, config = resolve_thinking_model("gpt-4:adaptive")
        assert base == "gpt-4"
        assert config == ADAPTIVE_THINKING_CONFIG


class TestApplyThinkingParams:

    def test_injects_thinking_config(self):
        body = {"model": "opus", "messages": []}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 32000}

    def test_bumps_max_tokens_when_too_low(self):
        body = {"model": "opus", "messages": [], "max_tokens": 100}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["max_tokens"] == 64000

    def test_preserves_max_tokens_when_sufficient(self):
        body = {"model": "opus", "messages": [], "max_tokens": 128000}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["max_tokens"] == 128000

    def test_sets_max_tokens_when_missing(self):
        body = {"model": "opus", "messages": []}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["max_tokens"] == 64000

    def test_adaptive_sets_min_max_tokens(self):
        body = {"model": "opus", "messages": []}
        result = apply_thinking_params(body, ThinkingConfig(type="adaptive"))
        assert result["max_tokens"] == 64000

    def test_haiku_smaller_min_max_tokens(self):
        body = {"model": "haiku", "messages": [], "max_tokens": 100}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=16000))
        assert result["max_tokens"] == 32000

    def test_does_not_mutate_original(self):
        body = {"model": "opus", "messages": []}
        apply_thinking_params(body, ThinkingConfig(type="adaptive"))
        assert "thinking" not in body

    def test_uses_max_completion_tokens_when_max_tokens_absent(self):
        body = {"model": "opus", "messages": [], "max_completion_tokens": 100}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["max_tokens"] == 64000
        assert result["max_completion_tokens"] == 100

    def test_max_tokens_zero_treated_as_unset(self):
        body = {"model": "opus", "messages": [], "max_tokens": 0}
        result = apply_thinking_params(body, ThinkingConfig(type="enabled", budget_tokens=32000))
        assert result["max_tokens"] == 64000
