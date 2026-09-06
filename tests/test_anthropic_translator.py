import json

from src.proxy.anthropic.translator import (
    StreamingState,
    create_anthropic_error,
    translate_request,
    translate_response,
)


class TestCreateAnthropicError:
    def test_basic_error(self):
        result = create_anthropic_error("bad request", "invalid_request_error")
        assert result["type"] == "error"
        assert result["error"]["type"] == "invalid_request_error"
        assert result["error"]["message"] == "bad request"

    def test_default_error_type(self):
        result = create_anthropic_error("something went wrong")
        assert result["error"]["type"] == "invalid_request_error"


class TestTranslateRequest:
    def test_basic_request(self):
        body = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        result = translate_request(body)
        assert result["model"] == "claude-sonnet-4-20250514"
        assert result["max_tokens"] == 1024
        assert result["messages"] == [{"role": "user", "content": "Hello"}]
        assert result["stream"] is False

    def test_system_string(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": "You are helpful.",
        }
        result = translate_request(body)
        assert result["messages"][0] == {"role": "system", "content": "You are helpful."}
        assert result["messages"][1] == {"role": "user", "content": "Hi"}

    def test_system_array(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "system": [
                {"type": "text", "text": "Part 1."},
                {"type": "text", "text": "Part 2."},
            ],
        }
        result = translate_request(body)
        assert result["messages"][0] == {"role": "system", "content": "Part 1.\n\nPart 2."}

    def test_messages_with_text_blocks(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }
        result = translate_request(body)
        assert result["messages"][0]["content"] == "Hello"

    def test_messages_with_image_base64(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": "abc123"},
                }],
            }],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_messages_with_image_url(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {"type": "url", "url": "https://example.com/img.png"},
                }],
            }],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/img.png"

    def test_messages_with_document(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "document",
                    "source": {"type": "base64", "media_type": "application/pdf", "data": "..."},
                    "title": "my_doc.pdf",
                }],
            }],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert "my_doc.pdf" in content

    def test_tool_use_in_assistant_message(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search."},
                    {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "test"}},
                ],
            }],
        }
        result = translate_request(body)
        msg = result["messages"][0]
        assert msg["content"] == "Let me search."
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["id"] == "toolu_1"
        assert msg["tool_calls"][0]["type"] == "function"
        assert msg["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}

    def test_tool_result_in_user_message(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": "result text"},
                ],
            }],
        }
        result = translate_request(body)
        assert result["messages"][0]["role"] == "tool"
        assert result["messages"][0]["tool_call_id"] == "toolu_1"
        assert result["messages"][0]["content"] == "result text"

    def test_tool_result_with_list_content(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "line 1"}, {"type": "text", "text": "line 2"}],
                    },
                ],
            }],
        }
        result = translate_request(body)
        assert result["messages"][0]["content"] == "line 1\nline 2"

    def test_thinking_blocks_in_assistant_message(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think...", "signature": "sig123"},
                    {"type": "text", "text": "Answer."},
                ],
            }],
        }
        result = translate_request(body)
        msg = result["messages"][0]
        assert msg["content"] == "Answer."

    def test_tools_translation(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
            }],
        }
        result = translate_request(body)
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"
        assert result["tools"][0]["function"]["description"] == "Get current weather"
        assert result["tools"][0]["function"]["parameters"]["properties"]["city"]["type"] == "string"

    def test_tool_choice_auto(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "auto"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "any"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == "required"

    def test_tool_choice_none(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "none"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == "none"

    def test_tool_choice_specific_tool(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "tool_choice": {"type": "tool", "name": "search"},
        }
        result = translate_request(body)
        assert result["tool_choice"] == {"type": "function", "function": {"name": "search"}}

    def test_thinking_config_passthrough(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = translate_request(body)
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_stop_sequences(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["\n\nHuman:"],
        }
        result = translate_request(body)
        assert result["stop"] == ["\n\nHuman:"]

    def test_temperature_and_top_p(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "top_p": 0.9,
        }
        result = translate_request(body)
        assert result["temperature"] == 0.7
        assert result["top_p"] == 0.9

    def test_top_k_passthrough(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "top_k": 40,
        }
        result = translate_request(body)
        assert result["top_k"] == 40

    def test_metadata_passthrough(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "metadata": {"user_id": "u123"},
        }
        result = translate_request(body)
        assert result["metadata"] == {"user_id": "u123"}

    def test_output_config_effort(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "output_config": {"effort": "high"},
        }
        result = translate_request(body)
        assert result["effort"] == "high"

    def test_output_config_json_schema(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hi"}],
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "json_schema": {"type": "object", "properties": {"name": {"type": "string"}}},
                },
            },
        }
        result = translate_request(body)
        assert result["response_format"]["type"] == "json_schema"
        assert result["response_format"]["json_schema"]["schema"]["properties"]["name"]["type"] == "string"

    def test_empty_content(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": []}],
        }
        result = translate_request(body)
        assert result["messages"][0]["content"] == []

    def test_multiple_content_blocks(self):
        body = {
            "model": "m",
            "max_tokens": 100,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "xyz"}},
                ],
            }],
        }
        result = translate_request(body)
        content = result["messages"][0]["content"]
        assert len(content) == 2
        assert content[0] == {"type": "text", "text": "Look at this:"}
        assert content[1]["type"] == "image_url"


class TestTranslateResponse:
    def test_basic_text_response(self):
        openai_resp = {
            "id": "chatcmpl-1",
            "model": "claude-sonnet-4-20250514",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = translate_response(openai_resp, "claude-sonnet-4-20250514")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["id"].startswith("msg_")
        assert result["content"] == [{"type": "text", "text": "Hello!"}]
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_calls_response(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10},
        }
        result = translate_response(openai_resp, "m")
        assert result["stop_reason"] == "tool_use"
        tool_block = result["content"][0]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "call_1"
        assert tool_block["name"] == "search"
        assert tool_block["input"] == {"q": "test"}

    def test_thinking_blocks_response(self):
        openai_resp = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Four.",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "2+2=4", "signature": "EpQC..."},
                    ],
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 20},
        }
        result = translate_response(openai_resp, "m")
        assert result["content"][0]["type"] == "thinking"
        assert result["content"][0]["thinking"] == "2+2=4"
        assert result["content"][0]["signature"] == "EpQC..."
        assert result["content"][1]["type"] == "text"
        assert result["content"][1]["text"] == "Four."

    def test_finish_reason_length(self):
        openai_resp = {
            "choices": [{"message": {"role": "assistant", "content": "..."}, "finish_reason": "length"}],
            "usage": {},
        }
        result = translate_response(openai_resp, "m")
        assert result["stop_reason"] == "max_tokens"

    def test_empty_content_response(self):
        openai_resp = {
            "choices": [{"message": {"role": "assistant", "content": None}, "finish_reason": "stop"}],
            "usage": {},
        }
        result = translate_response(openai_resp, "m")
        assert result["content"] == [{"type": "text", "text": ""}]

    def test_no_choices(self):
        openai_resp = {"choices": [], "usage": {}}
        result = translate_response(openai_resp, "m")
        assert result["content"] == [{"type": "text", "text": ""}]
        assert result["stop_reason"] is None


class TestStreamingState:
    def test_first_chunk_emits_message_start(self):
        state = StreamingState(model="claude-sonnet-4-20250514")
        chunk = {
            "id": "chatcmpl-1",
            "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
        }
        events = state.translate_chunk(chunk)
        assert events[0]["type"] == "message_start"
        assert events[0]["message"]["model"] == "claude-sonnet-4-20250514"

    def test_text_content_emits_block_start_and_delta(self):
        state = StreamingState(model="m")
        chunk = {
            "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}],
        }
        events = state.translate_chunk(chunk)
        type_sequence = [e["type"] for e in events]
        assert "content_block_start" in type_sequence
        assert "content_block_delta" in type_sequence
        start_evt = next(e for e in events if e["type"] == "content_block_start")
        assert start_evt["content_block"]["type"] == "text"
        delta_evt = next(e for e in events if e["type"] == "content_block_delta")
        assert delta_evt["delta"]["type"] == "text_delta"
        assert delta_evt["delta"]["text"] == "Hello"

    def test_reasoning_content_emits_thinking_block(self):
        state = StreamingState(model="m")
        chunk = {
            "choices": [{"index": 0, "delta": {"reasoning_content": "Let me think..."}, "finish_reason": None}],
        }
        events = state.translate_chunk(chunk)
        start_evt = next(e for e in events if e["type"] == "content_block_start")
        assert start_evt["content_block"]["type"] == "thinking"
        delta_evt = next(e for e in events if e["type"] == "content_block_delta")
        assert delta_evt["delta"]["type"] == "thinking_delta"
        assert delta_evt["delta"]["thinking"] == "Let me think..."

    def test_tool_call_emits_tool_use_block(self):
        state = StreamingState(model="m")
        state.started = True
        chunk = {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {"name": "search", "arguments": '{"q":'},
                    }],
                },
                "finish_reason": None,
            }],
        }
        events = state.translate_chunk(chunk)
        start_evt = next(e for e in events if e["type"] == "content_block_start")
        assert start_evt["content_block"]["type"] == "tool_use"
        assert start_evt["content_block"]["name"] == "search"
        delta_evt = next(e for e in events if e["type"] == "content_block_delta")
        assert delta_evt["delta"]["type"] == "input_json_delta"
        assert delta_evt["delta"]["partial_json"] == '{"q":'

    def test_finish_reason_defers_terminal_events_to_finalize(self):
        state = StreamingState(model="m")
        state.started = True
        state.current_block_type = "text"
        chunk = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        events = state.translate_chunk(chunk)
        type_sequence = [e["type"] for e in events]
        assert "message_delta" not in type_sequence
        assert "message_stop" not in type_sequence

        final = state.finalize()
        final_sequence = [e["type"] for e in final]
        assert final_sequence.index("content_block_stop") < final_sequence.index("message_delta")
        assert "message_stop" in final_sequence
        delta_evt = next(e for e in final if e["type"] == "message_delta")
        assert delta_evt["delta"]["stop_reason"] == "end_turn"

    def test_finalize_emits_closing_events(self):
        state = StreamingState(model="m")
        state.started = True
        state.current_block_type = "text"
        events = state.finalize()
        type_sequence = [e["type"] for e in events]
        assert "content_block_stop" in type_sequence
        assert "message_delta" in type_sequence
        assert "message_stop" in type_sequence

    def test_block_transition_closes_previous(self):
        state = StreamingState(model="m")
        state.started = True

        chunk1 = {
            "choices": [{"index": 0, "delta": {"reasoning_content": "think"}, "finish_reason": None}],
        }
        state.translate_chunk(chunk1)
        assert state.current_block_type == "thinking"

        chunk2 = {
            "choices": [{"index": 0, "delta": {"content": "answer"}, "finish_reason": None}],
        }
        events = state.translate_chunk(chunk2)
        type_sequence = [e["type"] for e in events]
        assert "content_block_stop" in type_sequence
        assert state.current_block_type == "text"
        assert state.block_index == 1


class TestStreamingStateLifecycle:
    def test_finalize_on_unstarted_state_emits_message_start_first(self):
        state = StreamingState(model="claude-sonnet-4-20250514")
        events = state.finalize()
        type_sequence = [e["type"] for e in events]
        assert type_sequence[0] == "message_start"
        assert type_sequence[-1] == "message_stop"
        assert type_sequence.count("message_start") == 1
        assert type_sequence.count("message_stop") == 1
        assert events[0]["message"]["model"] == "claude-sonnet-4-20250514"

    def test_finish_reason_then_finalize_emits_single_terminal_sequence(self):
        state = StreamingState(model="m")
        first = state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        })
        second = state.finalize()
        combined = [e["type"] for e in first + second]
        assert combined.count("message_start") == 1
        assert combined.count("message_delta") == 1
        assert combined.count("message_stop") == 1

    def test_finalize_is_idempotent(self):
        state = StreamingState(model="m")
        state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        })
        combined = [e["type"] for e in state.finalize() + state.finalize()]
        assert combined.count("message_delta") == 1
        assert combined.count("message_stop") == 1

    def test_message_stop_is_strictly_terminal(self):
        state = StreamingState(model="m")
        events = state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        }) + state.finalize()
        type_sequence = [e["type"] for e in events]
        assert type_sequence[-1] == "message_stop"


class TestStreamingStateUsage:
    def test_usage_arriving_after_finish_reason_is_reported(self):
        state = StreamingState(model="m")
        state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}],
        })
        state.translate_chunk({
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 42, "total_tokens": 53},
        })
        events = state.finalize()
        delta_evt = next(e for e in events if e["type"] == "message_delta")
        assert delta_evt["usage"]["output_tokens"] == 42

    def test_total_tokens_is_not_used_as_output_token_fallback(self):
        state = StreamingState(model="m")
        state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
        })
        state.translate_chunk({
            "choices": [],
            "usage": {"prompt_tokens": 100, "total_tokens": 130},
        })
        events = state.finalize()
        delta_evt = next(e for e in events if e["type"] == "message_delta")
        assert delta_evt["usage"]["output_tokens"] != 130

    def test_input_tokens_reported_on_message_start(self):
        state = StreamingState(model="m")
        events = state.translate_chunk({
            "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 0},
        })
        start_evt = next(e for e in events if e["type"] == "message_start")
        assert start_evt["message"]["usage"]["input_tokens"] == 7


class TestStreamingStateParallelToolCalls:
    def _two_tool_chunk(self):
        return {
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 0, "id": "call_a", "function": {"name": "get_weather", "arguments": ""}},
                        {"index": 1, "id": "call_b", "function": {"name": "get_time", "arguments": ""}},
                    ],
                },
                "finish_reason": None,
            }],
        }

    def test_parallel_tool_calls_get_distinct_block_indices(self):
        state = StreamingState(model="m")
        events = state.translate_chunk(self._two_tool_chunk())
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert len(starts) == 2
        assert starts[0]["content_block"]["name"] == "get_weather"
        assert starts[1]["content_block"]["name"] == "get_time"
        assert starts[0]["index"] != starts[1]["index"]

    def test_argument_deltas_carry_their_own_block_index(self):
        state = StreamingState(model="m")
        start_events = state.translate_chunk(self._two_tool_chunk())
        starts = {e["content_block"]["name"]: e["index"] for e in start_events
                  if e["type"] == "content_block_start"}

        events = state.translate_chunk({
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {"index": 1, "function": {"arguments": '{"city":"Paris"}'}},
                        {"index": 0, "function": {"arguments": '{"city":"Tokyo"}'}},
                    ],
                },
                "finish_reason": None,
            }],
        })
        by_json = {e["delta"]["partial_json"]: e["index"] for e in events
                   if e["type"] == "content_block_delta"}
        assert by_json['{"city":"Tokyo"}'] == starts["get_weather"]
        assert by_json['{"city":"Paris"}'] == starts["get_time"]

    def test_each_parallel_tool_block_is_closed_once(self):
        state = StreamingState(model="m")
        events = state.translate_chunk(self._two_tool_chunk())
        events += state.translate_chunk({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })
        events += state.finalize()
        stops = [e for e in events if e["type"] == "content_block_stop"]
        assert len(stops) == 2
        assert len({e["index"] for e in stops}) == 2

    def test_tool_call_stop_reason_maps_to_tool_use(self):
        state = StreamingState(model="m")
        state.translate_chunk(self._two_tool_chunk())
        state.translate_chunk({
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        })
        events = state.finalize()
        delta_evt = next(e for e in events if e["type"] == "message_delta")
        assert delta_evt["delta"]["stop_reason"] == "tool_use"


class TestStreamingStateThinkingSignature:
    def test_signature_delta_is_emitted_when_upstream_supplies_one(self):
        state = StreamingState(model="m")
        state.translate_chunk({
            "choices": [{"index": 0, "delta": {"reasoning_content": "step"}, "finish_reason": None}],
        })
        events = state.translate_chunk({
            "choices": [{
                "index": 0,
                "delta": {"thinking_blocks": [{"type": "thinking", "signature": "sig-abc"}]},
                "finish_reason": None,
            }],
        })
        sig_deltas = [e for e in events
                      if e["type"] == "content_block_delta" and e["delta"].get("type") == "signature_delta"]
        assert len(sig_deltas) == 1
        assert sig_deltas[0]["delta"]["signature"] == "sig-abc"

