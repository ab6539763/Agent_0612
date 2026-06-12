"""AnthropicProvider 测试（SDK 客户端全 Mock）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic
from src.llm.anthropic_provider import AnthropicProvider, _build_kwargs, _split_system
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionRequest,
    FinishReason,
    ToolCallRequest,
    ToolSpec,
)


class FakeClient:
    """最小 AsyncAnthropic 替身。"""

    def __init__(self, response: Any = None, events: list[Any] | None = None) -> None:
        self.last_kwargs: dict[str, Any] = {}

        async def create(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            if kwargs.get("stream"):
                async def iterate() -> AsyncIterator[Any]:
                    for event in events or []:
                        yield event

                return iterate()
            return response

        self.messages = SimpleNamespace(create=create)


def make_provider(fake: FakeClient) -> AnthropicProvider:
    return AnthropicProvider(api_key="ak-test", client=cast(AsyncAnthropic, fake))


def make_request() -> CompletionRequest:
    return CompletionRequest(
        model="anthropic:claude-sonnet-4-5",
        messages=(
            ChatMessage(role=ChatRole.SYSTEM, content="be terse"),
            ChatMessage(role=ChatRole.USER, content="hello"),
        ),
        tools=(
            ToolSpec(
                name="search",
                description="search docs",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    )


class TestWireFormat:
    def test_system_extracted(self) -> None:
        system, messages = _split_system(make_request().messages)
        assert system == "be terse"
        assert messages == [{"role": "user", "content": "hello"}]

    def test_tool_result_becomes_user_block(self) -> None:
        _, messages = _split_system(
            (
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content="checking",
                    tool_calls=(ToolCallRequest(id="t1", name="search", arguments={}),),
                ),
                ChatMessage(role=ChatRole.TOOL, content="found it", tool_call_id="t1"),
            )
        )
        assert messages[0]["role"] == "assistant"
        blocks = messages[0]["content"]
        assert blocks[0] == {"type": "text", "text": "checking"}
        assert blocks[1]["type"] == "tool_use"
        assert messages[1]["content"][0]["type"] == "tool_result"
        assert messages[1]["content"][0]["tool_use_id"] == "t1"

    def test_max_tokens_defaulted(self) -> None:
        kwargs = _build_kwargs(make_request())
        assert kwargs["max_tokens"] == 4096
        assert kwargs["model"] == "claude-sonnet-4-5"
        assert kwargs["tools"][0]["input_schema"] == {"type": "object", "properties": {}}


class TestComplete:
    async def test_parses_text_and_tool_use(self) -> None:
        response = SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="Let me search. "),
                SimpleNamespace(
                    type="tool_use", id="tu_1", name="search", input={"q": "k8s"}
                ),
            ],
            stop_reason="tool_use",
            usage=SimpleNamespace(input_tokens=15, output_tokens=7),
        )

        result = await make_provider(FakeClient(response=response)).complete(make_request())

        assert result.message.content == "Let me search. "
        assert result.message.tool_calls == (
            ToolCallRequest(id="tu_1", name="search", arguments={"q": "k8s"}),
        )
        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert result.usage.prompt_tokens == 15


class TestStream:
    async def test_event_translation(self) -> None:
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=12)),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="text"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="Hi "),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="there"),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(type="tool_use", id="tu_2", name="search"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="input_json_delta", partial_json='{"q":'),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(type="input_json_delta", partial_json=' "redis"}'),
            ),
            SimpleNamespace(type="content_block_stop", index=1),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=9),
            ),
            SimpleNamespace(type="message_stop"),
        ]

        received = [
            chunk
            async for chunk in make_provider(FakeClient(events=events)).stream(
                make_request()
            )
        ]

        assert "".join(c.content_delta for c in received) == "Hi there"
        tool_chunks = [c for c in received if c.tool_call is not None]
        assert tool_chunks[0].tool_call == ToolCallRequest(
            id="tu_2", name="search", arguments={"q": "redis"}
        )
        final = received[-1]
        assert final.finish_reason is FinishReason.TOOL_CALLS
        assert final.usage is not None
        assert final.usage.prompt_tokens == 12
        assert final.usage.completion_tokens == 9

    async def test_empty_tool_json_defaults_to_empty_object(self) -> None:
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=1)),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=0,
                content_block=SimpleNamespace(type="tool_use", id="tu_3", name="ping"),
            ),
            SimpleNamespace(type="content_block_stop", index=0),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage=SimpleNamespace(output_tokens=1),
            ),
        ]
        received = [
            c
            async for c in make_provider(FakeClient(events=events)).stream(make_request())
        ]
        tool_chunk = next(c for c in received if c.tool_call is not None)
        assert tool_chunk.tool_call is not None
        assert tool_chunk.tool_call.arguments == {}


class TestTemperatureClamp:
    def test_temperature_clamped_to_anthropic_range(self) -> None:
        request = CompletionRequest(
            model="anthropic:claude-sonnet-4-5",
            messages=(ChatMessage(role=ChatRole.USER, content="hi"),),
            temperature=1.7,
        )
        assert _build_kwargs(request)["temperature"] == pytest.approx(1.0)
