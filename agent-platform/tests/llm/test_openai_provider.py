"""OpenAIProvider 测试（SDK 客户端全 Mock）。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import openai
import pytest
from openai import AsyncOpenAI
from src.core.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from src.llm._openai_format import (
    StreamingToolCallAggregator,
    build_request_kwargs,
    messages_to_openai,
)
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionRequest,
    FinishReason,
    ToolCallRequest,
    ToolSpec,
)
from src.llm.openai_provider import OpenAIProvider, _translate_error


class FakeClient:
    """最小 AsyncOpenAI 替身。"""

    def __init__(self, response: Any = None, chunks: list[Any] | None = None) -> None:
        self.last_kwargs: dict[str, Any] = {}

        async def create(**kwargs: Any) -> Any:
            self.last_kwargs = kwargs
            if kwargs.get("stream"):
                async def iterate() -> AsyncIterator[Any]:
                    for chunk in chunks or []:
                        yield chunk

                return iterate()
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def make_provider(fake: FakeClient) -> OpenAIProvider:
    return OpenAIProvider(api_key="sk-test", client=cast(AsyncOpenAI, fake))


def request_with_tools() -> CompletionRequest:
    return CompletionRequest(
        model="openai:gpt-4o",
        messages=(
            ChatMessage(role=ChatRole.SYSTEM, content="be helpful"),
            ChatMessage(role=ChatRole.USER, content="weather in beijing?"),
        ),
        tools=(
            ToolSpec(
                name="get_weather",
                description="query weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
    )


class TestComplete:
    async def test_parses_text_response(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="hello", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        fake = FakeClient(response=response)

        result = await make_provider(fake).complete(request_with_tools())

        assert result.message.content == "hello"
        assert result.finish_reason is FinishReason.STOP
        assert result.usage.total_tokens == 15
        assert result.model == "openai:gpt-4o"
        # 模型名剥离平台前缀后传给 SDK
        assert fake.last_kwargs["model"] == "gpt-4o"
        assert fake.last_kwargs["tools"][0]["function"]["name"] == "get_weather"

    async def test_parses_tool_calls(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="get_weather",
                                    arguments='{"city": "beijing"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=8),
        )

        result = await make_provider(FakeClient(response=response)).complete(
            request_with_tools()
        )

        assert result.finish_reason is FinishReason.TOOL_CALLS
        assert result.message.tool_calls == (
            ToolCallRequest(id="call_1", name="get_weather", arguments={"city": "beijing"}),
        )

    async def test_invalid_tool_arguments_rejected(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[
                            SimpleNamespace(
                                id="call_1",
                                function=SimpleNamespace(
                                    name="get_weather", arguments="{broken json"
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        )
        with pytest.raises(LLMInvalidResponseError):
            await make_provider(FakeClient(response=response)).complete(
                request_with_tools()
            )


class TestStream:
    async def test_aggregates_text_and_tool_calls(self) -> None:
        def delta_chunk(content: str | None = None, tool_calls: Any = None,
                        finish: str | None = None) -> Any:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                        finish_reason=finish,
                    )
                ],
                usage=None,
            )

        chunks = [
            delta_chunk(content="Let me "),
            delta_chunk(content="check."),
            delta_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id="call_9",
                        function=SimpleNamespace(name="get_weather", arguments='{"ci'),
                    )
                ]
            ),
            delta_chunk(
                tool_calls=[
                    SimpleNamespace(
                        index=0,
                        id=None,
                        function=SimpleNamespace(name=None, arguments='ty": "sh"}'),
                    )
                ]
            ),
            delta_chunk(finish="tool_calls"),
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=30, completion_tokens=12),
            ),
        ]

        received = [
            chunk
            async for chunk in make_provider(FakeClient(chunks=chunks)).stream(
                request_with_tools()
            )
        ]

        text = "".join(c.content_delta for c in received)
        assert text == "Let me check."
        tool_chunks = [c for c in received if c.tool_call is not None]
        assert len(tool_chunks) == 1
        assert tool_chunks[0].tool_call == ToolCallRequest(
            id="call_9", name="get_weather", arguments={"city": "sh"}
        )
        final = received[-1]
        assert final.finish_reason is FinishReason.TOOL_CALLS
        assert final.usage is not None
        assert final.usage.total_tokens == 42


class TestErrorTranslation:
    @staticmethod
    def _response(status: int) -> httpx.Response:
        request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
        return httpx.Response(status, request=request)

    def test_timeout(self) -> None:
        exc = openai.APITimeoutError(
            request=httpx.Request("POST", "https://api.openai.com")
        )
        assert isinstance(_translate_error(exc), LLMTimeoutError)

    def test_rate_limit(self) -> None:
        exc = openai.RateLimitError("429", response=self._response(429), body=None)
        translated = _translate_error(exc)
        assert isinstance(translated, LLMRateLimitError)
        assert translated.retryable

    def test_authentication(self) -> None:
        exc = openai.AuthenticationError("401", response=self._response(401), body=None)
        translated = _translate_error(exc)
        assert isinstance(translated, LLMAuthenticationError)
        assert not translated.retryable

    def test_context_length(self) -> None:
        exc = openai.BadRequestError(
            "This model's maximum context length is 128000 tokens",
            response=self._response(400),
            body=None,
        )
        assert isinstance(_translate_error(exc), LLMContextLengthError)

    def test_server_error(self) -> None:
        exc = openai.InternalServerError("boom", response=self._response(500), body=None)
        translated = _translate_error(exc)
        assert isinstance(translated, LLMServerError)
        assert translated.retryable


class TestWireFormat:
    def test_tool_message_conversion(self) -> None:
        payload = messages_to_openai(
            (
                ChatMessage(
                    role=ChatRole.ASSISTANT,
                    tool_calls=(
                        ToolCallRequest(id="c1", name="f", arguments={"x": 1}),
                    ),
                ),
                ChatMessage(
                    role=ChatRole.TOOL, content="result", tool_call_id="c1", name="f"
                ),
            )
        )
        assert payload[0]["tool_calls"][0]["function"]["name"] == "f"
        assert json.loads(payload[0]["tool_calls"][0]["function"]["arguments"]) == {"x": 1}
        assert payload[0]["content"] is None
        assert payload[1] == {"role": "tool", "tool_call_id": "c1", "content": "result"}

    def test_stop_and_max_tokens_forwarded(self) -> None:
        request = CompletionRequest(
            model="openai:gpt-4o",
            messages=(ChatMessage(role=ChatRole.USER, content="hi"),),
            max_tokens=128,
            stop=("END",),
        )
        kwargs = build_request_kwargs(request)
        assert kwargs["max_tokens"] == 128
        assert kwargs["stop"] == ["END"]
        assert "tools" not in kwargs

    def test_aggregator_rejects_nameless_call(self) -> None:
        aggregator = StreamingToolCallAggregator()
        aggregator.feed(
            [SimpleNamespace(index=0, id=None, function=SimpleNamespace(name=None, arguments="{}"))]
        )
        with pytest.raises(LLMInvalidResponseError):
            aggregator.finalize()
