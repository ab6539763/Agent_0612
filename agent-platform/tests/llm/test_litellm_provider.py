"""LiteLLMProvider 测试（litellm.acompletion 全 Mock）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from src.core.exceptions import (
    LLMContextLengthError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.llm.base import ChatMessage, ChatRole, CompletionRequest, FinishReason
from src.llm.litellm_provider import LiteLLMProvider, _translate_error


def make_request(model: str = "vllm:qwen2.5-7b") -> CompletionRequest:
    return CompletionRequest(
        model=model, messages=(ChatMessage(role=ChatRole.USER, content="hi"),)
    )


def fake_response(content: str = "ok") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=4, completion_tokens=2),
    )


class TestComplete:
    async def test_vllm_model_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return fake_response("from-vllm")

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        provider = LiteLLMProvider(api_base="http://vllm:8000/v1", api_key="k")

        result = await provider.complete(make_request("vllm:qwen2.5-7b"))

        assert result.message.content == "from-vllm"
        assert captured["model"] == "hosted_vllm/qwen2.5-7b"
        assert captured["api_base"] == "http://vllm:8000/v1"
        assert captured["api_key"] == "k"

    async def test_error_translated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing(**kwargs: Any) -> Any:
            raise litellm.exceptions.Timeout(
                "timed out", model="qwen", llm_provider="hosted_vllm"
            )

        monkeypatch.setattr(litellm, "acompletion", failing)
        provider = LiteLLMProvider(api_base="http://vllm:8000/v1")

        with pytest.raises(LLMTimeoutError):
            await provider.complete(make_request())


class TestStream:
    async def test_stream_chunks(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def chunk(content: str | None, finish: str | None = None) -> Any:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content, tool_calls=None),
                        finish_reason=finish,
                    )
                ],
                usage=None,
            )

        async def fake_acompletion(**kwargs: Any) -> Any:
            async def iterate() -> AsyncIterator[Any]:
                yield chunk("he")
                yield chunk("llo", finish="stop")
                yield SimpleNamespace(
                    choices=[],
                    usage=SimpleNamespace(prompt_tokens=3, completion_tokens=2),
                )

            return iterate()

        monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
        provider = LiteLLMProvider(api_base="http://vllm:8000/v1")

        chunks = [c async for c in provider.stream(make_request())]

        assert "".join(c.content_delta for c in chunks) == "hello"
        assert chunks[-1].finish_reason is FinishReason.STOP
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 5


class TestErrorTranslation:
    def test_rate_limit(self) -> None:
        exc = litellm.exceptions.RateLimitError(
            "429", model="m", llm_provider="openai"
        )
        assert isinstance(_translate_error(exc), LLMRateLimitError)

    def test_context_window(self) -> None:
        exc = litellm.exceptions.ContextWindowExceededError(
            "too long", model="m", llm_provider="openai"
        )
        assert isinstance(_translate_error(exc), LLMContextLengthError)
