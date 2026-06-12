"""模型前缀路由器：按 ``<prefix>:<model>`` 选择 Provider。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from src.core.exceptions import ValidationError
from src.llm.base import (
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    LLMProvider,
)


class ProviderRouter:
    """:class:`~src.llm.base.LLMProvider` 的路由实现。

    模型名约定 ``<prefix>:<model>``（如 ``openai:gpt-4o`` / ``vllm:qwen2.5-72b``）；
    未知前缀或缺少前缀时拒绝请求（不做隐式默认，防错配静默走错厂商）。
    """

    def __init__(self, providers: dict[str, LLMProvider]) -> None:
        """初始化路由器。

        Args:
            providers: 前缀到 Provider 的映射（已套弹性装饰器的实例）。
        """
        self._providers = dict(providers)

    def register(self, prefix: str, provider: LLMProvider) -> None:
        """注册或替换一个前缀的 Provider。"""
        self._providers[prefix] = provider

    @property
    def prefixes(self) -> frozenset[str]:
        """已注册的前缀集合。"""
        return frozenset(self._providers)

    def _resolve(self, model: str) -> LLMProvider:
        prefix, separator, rest = model.partition(":")
        if not separator or not rest:
            raise ValidationError(
                "model must be qualified as '<provider>:<model>'",
                details={"model": model, "known_prefixes": sorted(self._providers)},
            )
        provider = self._providers.get(prefix)
        if provider is None:
            raise ValidationError(
                f"no provider registered for prefix '{prefix}'",
                details={"model": model, "known_prefixes": sorted(self._providers)},
            )
        return provider

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """路由后执行非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        return await self._resolve(request.model).complete(request)

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """路由后执行流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。"""
        async for chunk in self._resolve(request.model).stream(request):
            yield chunk
