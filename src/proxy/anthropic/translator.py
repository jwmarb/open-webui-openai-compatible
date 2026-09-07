"""Bidirectional translation between Anthropic Messages API and OpenAI chat format."""

from __future__ import annotations

import json
import uuid
from typing import Any

from .models import (
    AnthropicErrorDetail,
    AnthropicErrorResponse,
    AnthropicResponse,
    AnthropicUsage,
)

__all__ = [
    "create_anthropic_error",
    "translate_request",
    "translate_response",
    "StreamingState",
]


def create_anthropic_error(
    message: str,
    error_type: str = "invalid_request_error",
) -> dict[str, Any]:
    resp = AnthropicErrorResponse(error=AnthropicErrorDetail(type=error_type, message=message))
    return resp.model_dump()


def _translate_system(system: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    texts = [block.get("text", "") for block in system if block.get("type") == "text"]
    combined = "\n\n".join(texts)
    return [{"role": "system", "content": combined}]


def _translate_tool_choice(tool_choice: dict[str, Any] | None) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    tc_type = tool_choice.get("type")
    if tc_type == "auto":
        return "auto"
    if tc_type == "any":
        return "required"
    if tc_type == "none":
        return "none"
    if tc_type == "tool":
        return {"type": "function", "function": {"name": tool_choice.get("name", "")}}
    return None


def _translate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    result = []
    for tool in tools:
        result.append({
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return result


def _translate_content_blocks(
    content: str | list[dict[str, Any]],
    role: str,
) -> tuple[Any, list[dict[str, Any]] | None, list[dict[str, Any]]]:
    """Translate Anthropic content blocks to OpenAI format.

    Returns (content_for_message, tool_calls_or_none, extra_messages).
    """
    if isinstance(content, str):
        return content, None, []

    openai_content: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    extra_messages: list[dict[str, Any]] = []
    thinking_blocks: list[dict[str, Any]] = []

    for block in content:
        block_type = block.get("type")

        if block_type == "text":
            openai_content.append({"type": "text", "text": block.get("text", "")})

        elif block_type == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                data = source.get("data", "")
                openai_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{data}"},
                })
            elif source.get("type") == "url":
                openai_content.append({
                    "type": "image_url",
                    "image_url": {"url": source.get("url", "")},
                })

        elif block_type == "document":
            openai_content.append({"type": "text", "text": f"[Document: {block.get('title', 'untitled')}]"})

        elif block_type == "tool_use" and role == "assistant":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

        elif block_type == "tool_result" and role == "user":
            tc_content = block.get("content")
            if isinstance(tc_content, list):
                text_parts = [p.get("text", "") for p in tc_content if p.get("type") == "text"]
                tc_text = "\n".join(text_parts)
            elif isinstance(tc_content, str):
                tc_text = tc_content
            else:
                tc_text = ""
            extra_messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": tc_text,
            })

        elif block_type in ("thinking", "redacted_thinking") and role == "assistant":
            thinking_blocks.append(block)

    if role == "assistant" and not openai_content and not tool_calls and thinking_blocks:
        openai_content = [{"type": "text", "text": ""}]

    content_value: Any
    if tool_calls:
        if len(openai_content) == 1 and openai_content[0].get("type") == "text":
            content_value = openai_content[0].get("text", "")
        elif not openai_content:
            content_value = None
        else:
            content_value = openai_content
    elif len(openai_content) == 1 and openai_content[0].get("type") == "text":
        content_value = openai_content[0].get("text", "")
    elif openai_content:
        content_value = openai_content
    else:
        content_value = "" if role == "assistant" else openai_content

    return content_value, tool_calls or None, extra_messages


def _translate_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        content_value, tool_calls, extra_msgs = _translate_content_blocks(content, role)

        has_real_content = content_value not in (None, "", [])
        if has_real_content or tool_calls or not extra_msgs:
            openai_msg: dict[str, Any] = {"role": role, "content": content_value}
            if tool_calls:
                openai_msg["tool_calls"] = tool_calls
            result.append(openai_msg)
        result.extend(extra_msgs)

    return result


def translate_request(body: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic Messages API request to OpenAI chat format."""
    openai_body: dict[str, Any] = {}

    openai_body["model"] = body.get("model", "")
    openai_body["max_tokens"] = body.get("max_tokens", 4096)
    openai_body["stream"] = body.get("stream", False)

    system_msgs = _translate_system(body.get("system"))
    user_msgs = _translate_messages(body.get("messages", []))
    openai_body["messages"] = system_msgs + user_msgs

    if body.get("stop_sequences"):
        openai_body["stop"] = body["stop_sequences"]
    if body.get("temperature") is not None:
        openai_body["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        openai_body["top_p"] = body["top_p"]

    tools = _translate_tools(body.get("tools"))
    if tools:
        openai_body["tools"] = tools

    tc = _translate_tool_choice(body.get("tool_choice"))
    if tc is not None:
        openai_body["tool_choice"] = tc

    if body.get("thinking"):
        thinking = body["thinking"]
        if isinstance(thinking, dict):
            openai_body["thinking"] = thinking
        elif hasattr(thinking, "model_dump"):
            openai_body["thinking"] = thinking.model_dump(exclude_none=True)
        else:
            openai_body["thinking"] = thinking

    if body.get("top_k") is not None:
        openai_body["top_k"] = body["top_k"]

    if body.get("metadata"):
        metadata = body["metadata"]
        if hasattr(metadata, "model_dump"):
            openai_body["metadata"] = metadata.model_dump(exclude_none=True)
        else:
            openai_body["metadata"] = metadata

    if body.get("output_config"):
        oc = body["output_config"]
        if hasattr(oc, "model_dump"):
            oc = oc.model_dump(exclude_none=True)
        if oc.get("effort"):
            # Nested, not a bare top-level ``effort``. Every Bedrock model on the
            # gateway rejects the flat key with "effort: Extra inputs are not
            # permitted"; the nested form is what upstream documents when it
            # refuses thinking.type="enabled". Verified 2026-09-06.
            openai_body["output_config"] = {"effort": oc["effort"]}
        if oc.get("format"):
            fmt = oc["format"]
            if fmt.get("type") == "json_schema":
                openai_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"schema": fmt.get("json_schema", {})},
                }

    return openai_body


def _map_finish_reason(reason: str | None) -> str | None:
    if reason is None:
        return None
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    return mapping.get(reason, "end_turn")


def translate_response(openai_response: dict[str, Any], model: str) -> dict[str, Any]:
    """Translate an OpenAI chat completion response to Anthropic format."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    content: list[dict[str, Any]] = []

    choices = openai_response.get("choices", [])
    message = choices[0].get("message", {}) if choices else {}

    thinking_blocks = message.get("thinking_blocks", [])
    for tb in thinking_blocks:
        content.append({
            "type": "thinking",
            "thinking": tb.get("thinking", ""),
            "signature": tb.get("signature", ""),
        })

    msg_content = message.get("content")
    if msg_content:
        content.append({"type": "text", "text": msg_content})

    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except (json.JSONDecodeError, TypeError):
            input_data = {}
        content.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": func.get("name", ""),
            "input": input_data,
        })

    if not content:
        content.append({"type": "text", "text": ""})

    finish_reason = choices[0].get("finish_reason") if choices else None
    stop_reason = _map_finish_reason(finish_reason)

    usage = openai_response.get("usage", {})
    anthropic_usage = AnthropicUsage(
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )

    response = AnthropicResponse(
        id=msg_id,
        content=content,
        model=model,
        stop_reason=stop_reason,
        usage=anthropic_usage,
    )
    return response.model_dump()


class StreamingState:
    """Translates OpenAI streaming chunks to Anthropic SSE events.

    Owns the whole message lifecycle: exactly one `message_start`, block-index
    allocation (including one block per upstream tool-call index), usage
    collection, and exactly one terminal `message_delta`/`message_stop` pair
    emitted only from `finalize()`. Callers feed chunks and call `finalize()`
    once; they never infer ordering themselves.
    """

    def __init__(self, model: str, request_id: str | None = None):
        self.model = model
        self.msg_id = request_id or f"msg_{uuid.uuid4().hex[:24]}"
        self.block_index = 0
        self.current_block_type: str | None = None
        self.started = False
        self.input_tokens = 0
        self.output_tokens = 0
        self._tool_calls: dict[int, dict[str, str]] = {}
        self._stop_reason: str | None = None
        self._finalized = False

    def _start_message_event(self) -> dict[str, Any]:
        self.started = True
        return {
            "type": "message_start",
            "message": {
                "id": self.msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            },
        }

    def _content_block_start(self, block: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "type": "content_block_start",
            "index": index,
            "content_block": block,
        }

    def _content_block_delta(self, delta: dict[str, Any], index: int) -> dict[str, Any]:
        return {
            "type": "content_block_delta",
            "index": index,
            "delta": delta,
        }

    def _content_block_stop(self, index: int) -> dict[str, Any]:
        return {
            "type": "content_block_stop",
            "index": index,
        }

    def _close_current_block(self) -> list[dict[str, Any]]:
        if self.current_block_type is None:
            return []
        events = [self._content_block_stop(self.block_index)]
        self.block_index += 1
        self.current_block_type = None
        return events

    def _flush_tool_blocks(self) -> list[dict[str, Any]]:
        """Emit each buffered tool call as a complete, non-interleaved block.

        Anthropic requires a content block's index to equal its position in the
        final content array, and clients accumulate deltas per open block, so
        blocks must not overlap. Upstream interleaves tool-call fragments by
        index, so they are buffered until the stream ends and then emitted one
        block at a time.
        """
        events: list[dict[str, Any]] = []
        for tool_index in sorted(self._tool_calls):
            call = self._tool_calls[tool_index]
            events.append(self._content_block_start({
                "type": "tool_use",
                "id": call["id"],
                "name": call["name"],
                "input": {},
            }, self.block_index))
            if call["arguments"]:
                events.append(self._content_block_delta({
                    "type": "input_json_delta",
                    "partial_json": call["arguments"],
                }, self.block_index))
            events.append(self._content_block_stop(self.block_index))
            self.block_index += 1
        self._tool_calls.clear()
        return events

    def _absorb_usage(self, chunk: dict[str, Any]) -> None:
        usage = chunk.get("usage")
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens")
        if prompt_tokens is not None:
            self.input_tokens = prompt_tokens
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is not None:
            self.output_tokens = completion_tokens

    def translate_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        """Translate one OpenAI chunk. Terminal events are deferred to finalize()."""
        events: list[dict[str, Any]] = []

        self._absorb_usage(chunk)

        if not self.started:
            events.append(self._start_message_event())

        choices = chunk.get("choices", [])
        if not choices:
            return events

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        reasoning = delta.get("reasoning_content")
        if reasoning:
            if self.current_block_type != "thinking":
                events.extend(self._close_current_block())
                self.current_block_type = "thinking"
                events.append(self._content_block_start({
                    "type": "thinking",
                    "thinking": "",
                    "signature": "",
                }, self.block_index))
            events.append(self._content_block_delta({
                "type": "thinking_delta",
                "thinking": reasoning,
            }, self.block_index))

        for block in delta.get("thinking_blocks") or []:
            signature = block.get("signature")
            if signature and self.current_block_type == "thinking":
                events.append(self._content_block_delta({
                    "type": "signature_delta",
                    "signature": signature,
                }, self.block_index))

        content = delta.get("content")
        if content:
            if self.current_block_type != "text":
                events.extend(self._close_current_block())
                self.current_block_type = "text"
                events.append(self._content_block_start({
                    "type": "text",
                    "text": "",
                }, self.block_index))
            events.append(self._content_block_delta({
                "type": "text_delta",
                "text": content,
            }, self.block_index))

        tool_calls = delta.get("tool_calls")
        if tool_calls:
            self._buffer_tool_calls(tool_calls)

        if finish_reason:
            self._stop_reason = _map_finish_reason(finish_reason)

        return events

    def _buffer_tool_calls(self, tool_calls: list[dict[str, Any]]) -> None:
        for tc in tool_calls:
            tool_index = tc.get("index", 0)
            func = tc.get("function", {})
            call = self._tool_calls.setdefault(
                tool_index,
                {"id": "", "name": "", "arguments": ""},
            )
            if tc.get("id"):
                call["id"] = tc["id"]
            if func.get("name"):
                call["name"] = func["name"]
            call["arguments"] += func.get("arguments", "") or ""
            if not call["id"]:
                call["id"] = f"toolu_{uuid.uuid4().hex[:24]}"

    def finalize(self) -> list[dict[str, Any]]:
        """Emit the single terminal sequence. Safe to call more than once."""
        if self._finalized:
            return []
        self._finalized = True

        events: list[dict[str, Any]] = []
        if not self.started:
            events.append(self._start_message_event())
        events.extend(self._close_current_block())
        events.extend(self._flush_tool_blocks())
        events.append({
            "type": "message_delta",
            "delta": {
                "stop_reason": self._stop_reason or "end_turn",
                "stop_sequence": None,
            },
            "usage": {"output_tokens": self.output_tokens},
        })
        events.append({"type": "message_stop"})
        return events

