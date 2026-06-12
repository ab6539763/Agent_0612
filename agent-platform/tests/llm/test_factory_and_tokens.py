"""factory 装配与 token 计数测试。"""

from __future__ import annotations

import pytest
from src.core.config import LLMSettings
from src.core.exceptions import ConfigurationError
from src.llm.base import ChatMessage, ChatRole, ToolCallRequest
from src.llm.factory import build_embedding_provider, build_llm_router
from src.llm.tokens import TokenCounter


class TestFactory:
    def test_no_provider_configured_fails(self) -> None:
        with pytest.raises(ConfigurationError, match="no LLM provider"):
            build_llm_router(LLMSettings())

    def test_builds_configured_prefixes(self) -> None:
        settings = LLMSettings.model_validate(
            {
                "openai_api_key": "sk-test",
                "anthropic_api_key": "ak-test",
                "vllm_base_url": "http://vllm:8000/v1",
            }
        )
        router = build_llm_router(settings)
        assert router.prefixes == frozenset({"openai", "anthropic", "vllm"})

    def test_embedding_requires_openai_key(self) -> None:
        with pytest.raises(ConfigurationError, match="embedding"):
            build_embedding_provider(LLMSettings())

    def test_embedding_dimension_exposed(self) -> None:
        settings = LLMSettings.model_validate(
            {"openai_api_key": "sk-test", "embedding_dimension": 768}
        )
        assert build_embedding_provider(settings).dimension == 768


class TestTokenCounter:
    def test_text_count_positive(self) -> None:
        counter = TokenCounter()
        assert counter.count_text("hello world") > 0
        assert counter.count_text("") == 0

    def test_message_includes_overhead_and_tool_calls(self) -> None:
        counter = TokenCounter()
        plain = ChatMessage(role=ChatRole.USER, content="hi")
        with_tool = ChatMessage(
            role=ChatRole.ASSISTANT,
            content="hi",
            tool_calls=(
                ToolCallRequest(id="c", name="search", arguments={"q": "long query"}),
            ),
        )
        assert counter.count_message(with_tool) > counter.count_message(plain)

    def test_messages_sum(self) -> None:
        counter = TokenCounter()
        messages = [
            ChatMessage(role=ChatRole.USER, content="a"),
            ChatMessage(role=ChatRole.ASSISTANT, content="b"),
        ]
        assert counter.count_messages(messages) == sum(
            counter.count_message(m) for m in messages
        )
