#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Collapsible, Footer, Header, Markdown, Select, Static, TextArea

load_dotenv(Path(__file__).resolve().parent / ".env")

PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8000")

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_STREAM_TIMEOUT = httpx.Timeout(None, connect=10.0, pool=120.0)

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass
class StreamToken:
    thinking: str = ""
    content: str = ""


@dataclass
class ThinkingTagParser:
    """Handles <think>...</think> tags split across arbitrary chunk boundaries.

    Maintains a small buffer only when a partial tag candidate is in flight.
    Used as fallback when the upstream doesn't provide a structured
    `reasoning_content` field (e.g. Ollama, DeepSeek).
    """

    _inside_think: bool = False
    _buf: str = ""

    def feed(self, text: str) -> StreamToken:
        self._buf += text
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        while self._buf:
            if self._inside_think:
                close_idx = self._buf.find(_THINK_CLOSE)
                if close_idx == -1:
                    held = False
                    for plen in range(min(len(_THINK_CLOSE) - 1, len(self._buf)), 0, -1):
                        if _THINK_CLOSE.startswith(self._buf[-plen:]):
                            thinking_parts.append(self._buf[:-plen])
                            self._buf = self._buf[-plen:]
                            held = True
                            break
                    if not held:
                        thinking_parts.append(self._buf)
                        self._buf = ""
                    break
                else:
                    thinking_parts.append(self._buf[:close_idx])
                    self._buf = self._buf[close_idx + len(_THINK_CLOSE):]
                    self._inside_think = False
            else:
                open_idx = self._buf.find(_THINK_OPEN)
                if open_idx == -1:
                    held = False
                    for plen in range(min(len(_THINK_OPEN) - 1, len(self._buf)), 0, -1):
                        if _THINK_OPEN.startswith(self._buf[-plen:]):
                            content_parts.append(self._buf[:-plen])
                            self._buf = self._buf[-plen:]
                            held = True
                            break
                    if not held:
                        content_parts.append(self._buf)
                        self._buf = ""
                    break
                else:
                    content_parts.append(self._buf[:open_idx])
                    self._buf = self._buf[open_idx + len(_THINK_OPEN):]
                    self._inside_think = True

        return StreamToken(
            thinking="".join(thinking_parts),
            content="".join(content_parts),
        )

    def flush(self) -> StreamToken:
        leftover = self._buf
        self._buf = ""
        if self._inside_think:
            return StreamToken(thinking=leftover)
        return StreamToken(content=leftover)


async def fetch_models(base_url: str) -> list[tuple[str, str]]:
    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        resp = await client.get("/v1/models")
        resp.raise_for_status()
        data = resp.json().get("data", [])
        return [(m["id"], m["id"]) for m in data]


async def stream_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
):
    body = {"model": model, "messages": messages, "stream": True}
    tag_parser = ThinkingTagParser()
    async with httpx.AsyncClient(base_url=base_url, timeout=_STREAM_TIMEOUT) as client:
        async with client.stream("POST", "/v1/chat/completions", json=body) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""

                    if reasoning:
                        yield StreamToken(thinking=reasoning, content=content)
                    elif content:
                        token = tag_parser.feed(content)
                        if token.thinking or token.content:
                            yield token

            leftover = tag_parser.flush()
            if leftover.thinking or leftover.content:
                yield leftover


class PromptInput(TextArea):
    BINDINGS: list[Binding] = []

    class Submitted(Message):
        def __init__(self, text_area: PromptInput, value: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.value = value

    def __init__(self, **kwargs) -> None:
        super().__init__(
            language=None,
            soft_wrap=True,
            show_line_numbers=False,
            tab_behavior="indent",
            **kwargs,
        )

    def _submit(self) -> None:
        value = self.text.strip()
        if value:
            self.post_message(self.Submitted(self, value))

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter" and "shift" not in event.key:
            event.prevent_default()
            event.stop()
            self._submit()
            return
        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return
        await super()._on_key(event)


class ChatApp(App):
    TITLE = "Open WebUI Chat"

    CSS = """
    #sidebar {
        width: 32;
        dock: left;
        background: $surface;
        padding: 1;
    }
    #sidebar-title {
        text-style: bold;
        margin-bottom: 1;
    }
    #model-select {
        width: 100%;
        margin-bottom: 1;
    }
    #status {
        color: $text-muted;
        margin-top: 1;
    }
    #chat-area {
        width: 1fr;
    }
    #messages {
        height: 1fr;
        padding: 0 1;
    }
    #input-bar {
        dock: bottom;
        height: auto;
        padding: 1;
    }
    #prompt-input {
        height: 4;
        width: 1fr;
    }
    #send-hint {
        width: auto;
        padding: 1 1 0 0;
        color: $text-muted;
    }
    .msg-user {
        margin: 1 0 0 8;
        padding: 1 2;
        background: $primary-darken-2;
        color: $text;
    }
    .msg-assistant {
        margin: 1 8 0 0;
        padding: 1 2;
        background: $surface;
        color: $text;
    }
    .thinking-block {
        margin: 1 8 0 0;
        padding: 0;
        color: $text-muted;
    }
    .thinking-block Markdown {
        padding: 0 2;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+n", "new_chat", "New chat"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._model: str = ""
        self._messages: list[dict[str, str]] = []
        self._streaming = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Static("Models", id="sidebar-title")
                yield Select[str]([], prompt="Loading…", id="model-select")
                yield Static("Ready", id="status")
            with Vertical(id="chat-area"):
                with VerticalScroll(id="messages"):
                    yield Static(
                        "[dim]Select a model and start chatting.[/]",
                        id="placeholder",
                    )
                with Horizontal(id="input-bar"):
                    yield PromptInput(id="prompt-input")
                    yield Static("Shift+Enter\nfor newline", id="send-hint")
        yield Footer()

    async def on_mount(self) -> None:
        self._load_models()

    @work
    async def _load_models(self) -> None:
        status = self.query_one("#status", Static)
        select = self.query_one("#model-select", Select)
        status.update("Fetching models…")
        try:
            models = await fetch_models(PROXY_URL)
            if not models:
                status.update("[red]No models found[/]")
                return
            select.set_options(models)
            select.prompt = "Select model"
            status.update(f"{len(models)} models loaded")
        except Exception as exc:
            status.update(f"[red]Error: {exc}[/]")

    @on(Select.Changed, "#model-select")
    def _model_changed(self, event: Select.Changed) -> None:
        if event.value is not Select.BLANK:
            self._model = str(event.value)
            self.query_one("#status", Static).update(f"Model: {self._model}")

    @on(PromptInput.Submitted)
    async def _on_submit(self, event: PromptInput.Submitted) -> None:
        text = event.value.strip()
        if not text or self._streaming:
            return
        if not self._model:
            self.notify("Pick a model first.", severity="warning")
            return

        event.text_area.clear()

        for p in self.query("#placeholder"):
            p.remove()

        self._messages.append({"role": "user", "content": text})
        container = self.query_one("#messages", VerticalScroll)
        await container.mount(Static(
            f"[bold]You[/]\n{text}",
            classes="msg-user",
        ))

        container.scroll_end(animate=False)
        self._stream_response()

    @work
    async def _stream_response(self) -> None:
        self._streaming = True
        status = self.query_one("#status", Static)
        status.update("Streaming…")
        container = self.query_one("#messages", VerticalScroll)

        thinking_md: Markdown | None = None
        thinking_stream = None
        thinking_collapsible: Collapsible | None = None
        full_thinking = ""

        response_md = Markdown("", classes="msg-assistant")
        await container.mount(response_md)
        response_stream = Markdown.get_stream(response_md)

        full_response = ""
        try:
            async for token in stream_chat(PROXY_URL, self._model, self._messages):
                if token.thinking:
                    if thinking_collapsible is None:
                        thinking_collapsible = Collapsible(
                            title="Thinking…",
                            collapsed=True,
                            classes="thinking-block",
                        )
                        thinking_md = Markdown("")
                        thinking_collapsible.compose_add_child(thinking_md)
                        await container.mount(thinking_collapsible, before=response_md)
                        thinking_stream = Markdown.get_stream(thinking_md)

                    full_thinking += token.thinking
                    if thinking_stream is not None:
                        await thinking_stream.write(token.thinking)

                if token.content:
                    full_response += token.content
                    await response_stream.write(token.content)

                container.scroll_end(animate=False)

        except httpx.HTTPStatusError as exc:
            error_text = f"\n\n**Error:** Upstream returned {exc.response.status_code}"
            full_response += error_text
            await response_stream.write(error_text)
        except httpx.ConnectError:
            error_text = f"\n\n**Error:** Cannot connect to proxy at `{PROXY_URL}`"
            full_response += error_text
            await response_stream.write(error_text)
        except Exception as exc:
            error_text = f"\n\n**Error:** {type(exc).__name__}: {exc}"
            full_response += error_text
            await response_stream.write(error_text)
        finally:
            if thinking_stream is not None:
                await thinking_stream.stop()
            await response_stream.stop()
            self._streaming = False

        if thinking_collapsible is not None:
            thinking_collapsible.title = f"Thinking ({len(full_thinking)} chars)"

        if full_response:
            self._messages.append({"role": "assistant", "content": full_response})

        status.update(f"Model: {self._model}")

    def action_new_chat(self) -> None:
        self._messages.clear()
        container = self.query_one("#messages", VerticalScroll)
        container.remove_children()
        container.mount(Static(
            "[dim]Select a model and start chatting.[/]",
            id="placeholder",
        ))
        self.query_one("#status", Static).update(f"Model: {self._model}")


def main() -> None:
    app = ChatApp()
    app.run()


if __name__ == "__main__":
    main()
