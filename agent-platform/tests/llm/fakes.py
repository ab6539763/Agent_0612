"""llm 测试用的 Fake Provider。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.core.exceptions import AgentPlatformError
from src.core.types import TokenUsage
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    FinishReason,
)


def make_request(model: str = "openai:gpt-4o") -> CompletionRequest:
    """构造最小补全请求。"""
    return CompletionRequest(
        model=model,
        messages=(ChatMessage(role=ChatRole.USER, content="hi"),)
    )


def make_result(content: str = "ok") -> CompletionResult:
    """构造最小补全结果。"""
    return CompletionResult(
        message=ChatMessage(role=ChatRole.ASSISTANT, content=content),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1),
        model="openai:gpt-4o",
    )


class ScriptedProvider:
    """按脚本依次返回结果或抛异常的 Fake Provider。

    ``script`` 中的元素：异常实例（抛出）或 CompletionResult（返回）。
    """

    def __init__(self, script: list[AgentPlatformError | CompletionResult]) -> None:
        self._script = list(script)
        self.calls = 0

    def _next(self) -> CompletionResult:
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, AgentPlatformError):
            raise item
        return item

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        return self._next()

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        result = self._next()  # 异常在产出首 chunk 前抛出
        yield CompletionChunk(content_delta=result.message.content)
        yield CompletionChunk(finish_reason=result.finish_reason, usage=result.usage)


class MidStreamFailingProvider:
    """先产出一个 chunk 再失败的 Provider（验证流中途不重试）。"""

    def __init__(self, error: AgentPlatformError) -> None:
        self._error = error
        self.calls = 0

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        raise NotImplementedError

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        self.calls += 1
        yield CompletionChunk(content_delta="partial")
        raise self._error
