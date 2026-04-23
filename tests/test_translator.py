from src.translator import translate_models_response, create_openai_error, sanitize_chat_body


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
        assert len(result["data"]) == 3
        assert result["data"][0]["id"] == "gpt-4"
        assert result["data"][1]["created"] == 0
        assert result["data"][2]["created"] == 0
        assert result["data"][2]["owned_by"] == ""

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
