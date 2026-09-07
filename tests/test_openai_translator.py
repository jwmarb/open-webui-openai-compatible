from src.models import OpenAIModel, ThinkingConfig
from src.translator import (
    apply_thinking_params,
    create_openai_error,
    generate_thinking_variants,
    resolve_thinking_model,
    rewrite_chat_body,
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
        assert "claude-3:extended" not in ids
        assert "claude-3:adaptive" not in ids
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
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "auto",
        }
        result = sanitize_chat_body(body)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["tool_choice"] == "auto"
        assert len(result["tools"]) == 1
        assert result["stream_options"] == {"include_usage": True}

    def test_passes_through_extra_fields(self):
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
        assert result["model"] == "gpt-4"
        assert result["extra_body"] == {}
        assert result["api_base"] == "http://example.com"
        assert result["custom_llm_provider"] == "openai"

    def test_empty_body(self):
        result = sanitize_chat_body({})
        assert "chat_id" in result
        assert result["chat_id"].startswith("local:")

    def test_preserves_most_openai_params(self):
        body = {
            "model": "gpt-4",
            "messages": [],
            "stream": True,
            "stream_options": {"include_usage": False},
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
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "response_format": {"type": "json_object"},
            "seed": 42,
            "service_tier": "default",
            "metadata": {},
            "store": True,
            "reasoning_effort": "medium",
        }
        result = sanitize_chat_body(body)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.5
        assert result["reasoning_effort"] == "medium"
        assert result["stream_options"] == {"include_usage": True}
        assert "functions" not in result
        assert "function_call" not in result

    def test_unknown_fields_passed_through(self):
        body = {"extra_body": {}, "api_base": "http://x", "custom_llm_provider": "openai", "litellm_call_id": "abc"}
        result = sanitize_chat_body(body)
        assert result["extra_body"] == {}
        assert result["api_base"] == "http://x"

    def test_strips_legacy_function_calling(self):
        body = {
            "model": "gpt-4",
            "messages": [],
            "functions": [{"name": "f"}],
            "function_call": "auto",
        }
        result = sanitize_chat_body(body)
        assert "functions" not in result
        assert "function_call" not in result


class TestStripIncompatibleThinking:
    def test_strips_thinking_for_openai_model(self):
        body = {
            "model": "openai.gpt-5.6-luna",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = rewrite_chat_body(body)
        assert "thinking" not in result

    def test_preserves_thinking_for_claude_model(self):
        body = {
            "model": "anthropic.claude-sonnet-4-6",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_strips_thinking_for_unknown_model(self):
        body = {"model": "meta.llama3", "messages": [], "thinking": {"type": "enabled"}}
        result = rewrite_chat_body(body)
        assert "thinking" not in result

    def test_preserves_reasoning_effort_for_openai_model(self):
        body = {"model": "openai.gpt-5.4-nano", "messages": [], "reasoning_effort": "high"}
        result = rewrite_chat_body(body)
        assert result["reasoning_effort"] == "high"

    def test_strips_reasoning_effort_for_gpt_5_6(self):
        """Upstream maps reasoning_effort onto Bedrock's Anthropic-only ``thinking``
        param for the gpt-5.6 line, so the request dies with
        400 ``unknown_parameter: 'thinking'`` even though the proxy never sent it.
        """
        for model in ("openai.gpt-5.6-sol", "openai.gpt-5.6-terra", "openai.gpt-5.6-luna"):
            body = {"model": model, "messages": [], "reasoning_effort": "xhigh"}
            result = rewrite_chat_body(body)
            assert "reasoning_effort" not in result, model

    def test_preserves_reasoning_effort_for_gpt_oss(self):
        body = {"model": "openai.gpt-oss-120b-1:0", "messages": [], "reasoning_effort": "high"}
        result = rewrite_chat_body(body)
        assert result["reasoning_effort"] == "high"

    def test_preserves_reasoning_effort_for_claude(self):
        body = {"model": "bedrock-claude-5-opus", "messages": [], "reasoning_effort": "high"}
        result = rewrite_chat_body(body)
        assert result["reasoning_effort"] == "high"


class TestStripIncompatibleEffortConfig:
    def test_strips_effort_for_claude_4_5(self):
        """Claude 4.5 and earlier answer output_config.effort with
        400 "This model does not support the effort parameter."
        """
        body = {
            "model": "bedrock-claude-4-5-haiku",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        result = rewrite_chat_body(body)
        assert "output_config" not in result

    def test_preserves_effort_for_claude_4_6(self):
        body = {
            "model": "bedrock-claude-4-6-sonnet",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        result = rewrite_chat_body(body)
        assert result["output_config"] == {"effort": "high"}

    def test_preserves_effort_for_claude_5(self):
        body = {
            "model": "bedrock-claude-5-sonnet",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        result = rewrite_chat_body(body)
        assert result["output_config"] == {"effort": "high"}

    def test_sibling_format_survives_the_effort_strip(self):
        """``format`` is not an effort control; it becomes ``response_format``."""
        body = {
            "model": "bedrock-claude-4-5-haiku",
            "messages": [],
            "output_config": {"effort": "high", "format": {"type": "json_schema"}},
        }
        result = rewrite_chat_body(body)
        assert result["output_config"] == {"format": {"type": "json_schema"}}

    def test_strips_output_config_wholesale_for_gpt_5_6(self):
        body = {
            "model": "openai.gpt-5.6-sol",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        result = rewrite_chat_body(body)
        assert "output_config" not in result

    def test_non_anthropic_models_keep_output_config(self):
        body = {
            "model": "openai.gpt-oss-120b-1:0",
            "messages": [],
            "output_config": {"effort": "high"},
        }
        result = rewrite_chat_body(body)
        assert result["output_config"] == {"effort": "high"}

    def test_reasoning_effort_strip_does_not_mutate_original(self):
        body = {"model": "openai.gpt-5.6-sol", "messages": [], "reasoning_effort": "high"}
        rewrite_chat_body(body)
        assert body["reasoning_effort"] == "high"

    def test_strips_all_reasoning_controls_for_gpt_5_6(self):
        body = {
            "model": "openai.gpt-5.6-sol",
            "messages": [],
            "reasoning_effort": "xhigh",
            "reasoning": {"effort": "high"},
            "effort": "high",
            "verbosity": "low",
            "textVerbosity": "low",
            "thinking": {"type": "enabled"},
        }
        result = rewrite_chat_body(body)
        for field in ("reasoning_effort", "reasoning", "effort", "verbosity", "textVerbosity", "thinking"):
            assert field not in result, field
        assert result["model"] == "openai.gpt-5.6-sol"

    def test_preserves_reasoning_controls_for_claude(self):
        body = {
            "model": "bedrock-claude-5-opus",
            "messages": [],
            "reasoning_effort": "high",
            "verbosity": "low",
        }
        result = rewrite_chat_body(body)
        assert result["reasoning_effort"] == "high"
        assert result["verbosity"] == "low"

    def test_preserves_reasoning_controls_for_gpt_oss(self):
        body = {
            "model": "openai.gpt-oss-120b-1:0",
            "messages": [],
            "reasoning_effort": "high",
            "verbosity": "low",
        }
        result = rewrite_chat_body(body)
        assert result["reasoning_effort"] == "high"
        assert result["verbosity"] == "low"

    def test_gpt_5_6_strip_leaves_core_params_intact(self):
        body = {
            "model": "openai.gpt-5.6-terra",
            "messages": [{"role": "user", "content": "hi"}],
            "max_completion_tokens": 2000,
            "reasoning_effort": "xhigh",
            "stream": True,
            "temperature": 0.5,
        }
        result = rewrite_chat_body(body)
        assert "reasoning_effort" not in result
        assert result["max_completion_tokens"] == 2000
        assert result["stream"] is True
        assert result["temperature"] == 0.5
        assert result["messages"] == [{"role": "user", "content": "hi"}]

    def test_does_not_mutate_original(self):
        body = {"model": "gpt-4o", "messages": [], "thinking": {"type": "enabled"}}
        rewrite_chat_body(body)
        assert body["thinking"] == {"type": "enabled"}

    def test_noop_when_thinking_absent(self):
        body = {"model": "openai.gpt-5.6-luna", "messages": []}
        result = rewrite_chat_body(body)
        assert "thinking" not in result

    def test_coerces_enabled_to_adaptive_for_adaptive_only_model(self):
        body = {
            "model": "bedrock-claude-5-opus",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "adaptive"}

    def test_coerces_enabled_to_adaptive_for_adaptive_only_sonnet(self):
        body = {
            "model": "bedrock-claude-5-sonnet",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 32000},
        }
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "adaptive"}

    def test_preserves_adaptive_for_adaptive_only_model(self):
        body = {"model": "bedrock-claude-5-opus", "messages": [], "thinking": {"type": "adaptive"}}
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "adaptive"}

    def test_does_not_mutate_original_on_coercion(self):
        body = {
            "model": "bedrock-claude-5-opus",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        rewrite_chat_body(body)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_preserves_enabled_for_claude_4_6(self):
        body = {
            "model": "bedrock-claude-4-6-opus",
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_coerces_enabled_for_fable(self):
        body = {"model": "claude-fable-5", "messages": [], "thinking": {"type": "enabled"}}
        result = rewrite_chat_body(body)
        assert result["thinking"] == {"type": "adaptive"}

    def test_leaves_non_dict_thinking_untouched_for_claude(self):
        body = {"model": "bedrock-claude-5-opus", "messages": [], "thinking": "enabled"}
        result = rewrite_chat_body(body)
        assert result["thinking"] == "enabled"


class TestGenerateThinkingVariants:
    def test_non_claude_model_no_variants(self):
        model = OpenAIModel(id="gpt-4", owned_by="OpenAI")
        assert generate_thinking_variants(model) == []

    def test_claude_opus_gets_adaptive_only(self):
        model = OpenAIModel(id="bedrock-claude-4-6-opus")
        assert [v.id for v in generate_thinking_variants(model)] == ["bedrock-claude-4-6-opus:adaptive"]

    def test_claude_4_6_sonnet_gets_adaptive_only(self):
        model = OpenAIModel(id="bedrock-claude-4-6-sonnet")
        assert [v.id for v in generate_thinking_variants(model)] == ["bedrock-claude-4-6-sonnet:adaptive"]

    def test_claude_5_opus_gets_adaptive(self):
        model = OpenAIModel(id="bedrock-claude-5-opus")
        assert [v.id for v in generate_thinking_variants(model)] == ["bedrock-claude-5-opus:adaptive"]

    def test_claude_4_5_sonnet_gets_no_variants(self):
        model = OpenAIModel(id="bedrock-claude-4-5-sonnet")
        assert generate_thinking_variants(model) == []

    def test_claude_haiku_gets_no_variants(self):
        model = OpenAIModel(id="bedrock-claude-4-5-haiku")
        assert generate_thinking_variants(model) == []

    def test_no_extended_variant_is_ever_offered(self):
        for model_id in (
            "bedrock-claude-4-6-opus",
            "bedrock-claude-5-sonnet",
            "bedrock-claude-4-5-haiku",
            "claude-fable-5",
        ):
            ids = [v.id for v in generate_thinking_variants(OpenAIModel(id=model_id))]
            assert not any(i.endswith(":extended") for i in ids)

    def test_variant_preserves_created_and_owned_by(self):
        model = OpenAIModel(id="bedrock-claude-4-6-opus", created=1700000000, owned_by="Anthropic")
        for v in generate_thinking_variants(model):
            assert v.created == 1700000000
            assert v.owned_by == "Anthropic"
            assert v.object == "model"

    def test_models_response_includes_variants(self):
        raw = {
            "data": [
                {"id": "bedrock-claude-4-6-opus", "owned_by": ""},
                {"id": "bedrock-claude-4-5-haiku", "owned_by": ""},
                {"id": "gpt-4", "owned_by": "OpenAI"},
            ]
        }
        ids = [m["id"] for m in translate_models_response(raw)["data"]]
        assert ids == [
            "bedrock-claude-4-6-opus",
            "bedrock-claude-4-6-opus:adaptive",
            "bedrock-claude-4-5-haiku",
            "gpt-4",
        ]


class TestResolveThinkingModel:
    def test_plain_model_no_thinking(self):
        base, config = resolve_thinking_model("bedrock-claude-4-6-opus")
        assert base == "bedrock-claude-4-6-opus"
        assert config is None

    def test_adaptive_suffix_stripped(self):
        base, config = resolve_thinking_model("bedrock-claude-4-6-sonnet:adaptive")
        assert base == "bedrock-claude-4-6-sonnet"
        assert config == ThinkingConfig(type="adaptive")

    def test_adaptive_only_model_adaptive_suffix_works(self):
        base, config = resolve_thinking_model("bedrock-claude-5-opus:adaptive")
        assert base == "bedrock-claude-5-opus"
        assert config == ThinkingConfig(type="adaptive")

    def test_fable_adaptive_works(self):
        base, config = resolve_thinking_model("claude-fable-5:adaptive")
        assert base == "claude-fable-5"
        assert config == ThinkingConfig(type="adaptive")

    def test_extended_suffix_is_not_recognised_and_passes_through(self):
        base, config = resolve_thinking_model("bedrock-claude-4-6-opus:extended")
        assert base == "bedrock-claude-4-6-opus:extended"
        assert config is None

    def test_adaptive_on_unsupported_model_strips_suffix_without_thinking(self):
        base, config = resolve_thinking_model("bedrock-claude-4-5-haiku:adaptive")
        assert base == "bedrock-claude-4-5-haiku"
        assert config is None

    def test_adaptive_on_claude_4_5_strips_suffix_without_thinking(self):
        base, config = resolve_thinking_model("bedrock-claude-4-5-sonnet:adaptive")
        assert base == "bedrock-claude-4-5-sonnet"
        assert config is None

    def test_non_claude_adaptive_suffix_yields_no_thinking(self):
        base, config = resolve_thinking_model("gpt-4:adaptive")
        assert base == "gpt-4"
        assert config is None

    def test_openai_model_adaptive_suffix_yields_no_thinking(self):
        base, config = resolve_thinking_model("openai.gpt-5.6-luna:adaptive")
        assert base == "openai.gpt-5.6-luna"
        assert config is None


class TestApplyThinkingParams:
    def test_injects_thinking_config(self):
        body = apply_thinking_params({"model": "m"}, ThinkingConfig(type="adaptive"))
        assert body["thinking"] == {"type": "adaptive"}

    def test_bumps_max_tokens_when_too_low(self):
        body = apply_thinking_params({"model": "m", "max_tokens": 100}, ThinkingConfig(type="adaptive"))
        assert body["max_tokens"] == 64000

    def test_preserves_max_tokens_when_sufficient(self):
        body = apply_thinking_params({"model": "m", "max_tokens": 100000}, ThinkingConfig(type="adaptive"))
        assert body["max_tokens"] == 100000

    def test_sets_max_tokens_when_missing(self):
        body = apply_thinking_params({"model": "m"}, ThinkingConfig(type="adaptive"))
        assert body["max_tokens"] == 64000

    def test_does_not_mutate_original(self):
        original = {"model": "m", "max_tokens": 10}
        apply_thinking_params(original, ThinkingConfig(type="adaptive"))
        assert original == {"model": "m", "max_tokens": 10}

    def test_respects_max_completion_tokens(self):
        body = apply_thinking_params(
            {"model": "m", "max_completion_tokens": 100000}, ThinkingConfig(type="adaptive"))
        assert "max_tokens" not in body or body["max_tokens"] == 100000


class TestBedrockToolScrubbing:
    def test_empty_tools_list_removed(self):
        body = {"model": "m", "messages": [], "tools": [], "tool_choice": "auto", "parallel_tool_calls": True}
        result = rewrite_chat_body(body)
        assert "tools" not in result
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result

    def test_dummy_tool_injected_when_messages_reference_tools(self):
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [{"id": "tc1", "function": {"name": "f"}}]},
                {"role": "tool", "tool_call_id": "tc1", "content": "result"},
            ],
            "tools": [],
        }
        result = rewrite_chat_body(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "dummy_tool"
        assert "tool_choice" not in result

    def test_no_dummy_tool_when_messages_clean(self):
        body = {
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        }
        result = rewrite_chat_body(body)
        assert "tools" not in result

    def test_tool_choice_none_strips_tools(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none",
        }
        result = rewrite_chat_body(body)
        assert "tools" not in result
        assert "tool_choice" not in result
        assert "parallel_tool_calls" not in result

    def test_tool_choice_any_coerced_to_auto(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "any",
        }
        result = rewrite_chat_body(body)
        assert result["tool_choice"] == "auto"
        assert len(result["tools"]) == 1

    def test_tool_choice_required_coerced_to_auto(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "required",
        }
        result = rewrite_chat_body(body)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_dict_type_none_strips_tools(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": {"type": "none"},
        }
        result = rewrite_chat_body(body)
        assert "tools" not in result
        assert "tool_choice" not in result

    def test_tool_choice_dict_type_any_coerced(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": {"type": "any"},
        }
        result = rewrite_chat_body(body)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_auto_preserved(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "auto",
        }
        result = rewrite_chat_body(body)
        assert result["tool_choice"] == "auto"
        assert len(result["tools"]) == 1

    def test_legacy_functions_stripped(self):
        body = {
            "model": "m",
            "messages": [],
            "functions": [{"name": "f"}],
            "function_call": "auto",
        }
        result = rewrite_chat_body(body)
        assert "functions" not in result
        assert "function_call" not in result

    def test_messages_with_tool_call_id_detected(self):
        body = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "tool", "tool_call_id": "tc1", "content": "res"},
            ],
        }
        result = rewrite_chat_body(body)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "dummy_tool"

    def test_parallel_tool_calls_stripped_with_tool_choice_none(self):
        body = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "tool_choice": "none",
            "parallel_tool_calls": True,
        }
        result = rewrite_chat_body(body)
        assert "parallel_tool_calls" not in result


class TestStreamUsageInjection:
    def test_stream_true_injects_include_usage(self):
        body = {"model": "m", "messages": [], "stream": True}
        result = rewrite_chat_body(body)
        assert result["stream_options"] == {"include_usage": True}

    def test_stream_true_overrides_include_usage_false(self):
        body = {"model": "m", "messages": [], "stream": True, "stream_options": {"include_usage": False}}
        result = rewrite_chat_body(body)
        assert result["stream_options"]["include_usage"] is True

    def test_stream_true_preserves_other_stream_options(self):
        body = {"model": "m", "messages": [], "stream": True, "stream_options": {"other_opt": "val"}}
        result = rewrite_chat_body(body)
        assert result["stream_options"] == {"include_usage": True, "other_opt": "val"}

    def test_stream_false_no_injection(self):
        body = {"model": "m", "messages": [], "stream": False}
        result = rewrite_chat_body(body)
        assert "stream_options" not in result

    def test_no_stream_key_no_injection(self):
        body = {"model": "m", "messages": []}
        result = rewrite_chat_body(body)
        assert "stream_options" not in result

    def test_stream_true_already_has_include_usage_true(self):
        body = {"model": "m", "messages": [], "stream": True, "stream_options": {"include_usage": True}}
        result = rewrite_chat_body(body)
        assert result["stream_options"] == {"include_usage": True}


class TestStripUnsupportedFields:
    def test_vector_store_ids_stripped(self):
        body = {"model": "m", "messages": [], "vector_store_ids": ["vs_abc"]}
        result = rewrite_chat_body(body)
        assert "vector_store_ids" not in result

    def test_file_ids_stripped(self):
        body = {"model": "m", "messages": [], "file_ids": ["file-123"]}
        result = rewrite_chat_body(body)
        assert "file_ids" not in result

    def test_unsupported_fields_stripped_with_other_fields_preserved(self):
        body = {"model": "m", "messages": [], "vector_store_ids": ["vs_abc"], "temperature": 0.5}
        result = rewrite_chat_body(body)
        assert "vector_store_ids" not in result
        assert result["temperature"] == 0.5


class TestRewriteChatBodyAlias:
    def test_sanitize_chat_body_is_alias_for_rewrite(self):
        assert sanitize_chat_body is rewrite_chat_body


class TestChatIdInjection:
    def test_injects_chat_id_with_local_prefix(self):
        body = {"model": "m", "messages": []}
        result = rewrite_chat_body(body)
        assert "chat_id" in result
        assert result["chat_id"].startswith("local:")

    def test_chat_id_is_uuid_format(self):
        import uuid
        body = {"model": "m", "messages": []}
        result = rewrite_chat_body(body)
        suffix = result["chat_id"].removeprefix("local:")
        uuid.UUID(suffix, version=4)

    def test_each_call_produces_unique_chat_id(self):
        body = {"model": "m", "messages": []}
        r1 = rewrite_chat_body(body)
        r2 = rewrite_chat_body(body)
        assert r1["chat_id"] != r2["chat_id"]

    def test_chat_id_not_in_original_body(self):
        body = {"model": "m", "messages": []}
        rewrite_chat_body(body)
        assert "chat_id" not in body


class TestSessionIdStripping:
    def test_session_id_stripped(self):
        body = {"model": "m", "messages": [], "session_id": "ws-123"}
        result = rewrite_chat_body(body)
        assert "session_id" not in result

    def test_no_session_id_no_error(self):
        body = {"model": "m", "messages": []}
        result = rewrite_chat_body(body)
        assert "session_id" not in result

    def test_session_id_not_in_original_body(self):
        body = {"model": "m", "messages": [], "session_id": "ws-123"}
        rewrite_chat_body(body)
        assert body["session_id"] == "ws-123"
