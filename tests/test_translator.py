from src.translator import translate_models_response, create_openai_error


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
