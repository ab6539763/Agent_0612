"""OpenAI 官方 async SDK 适配器（含 Embedding）。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import openai
from openai import AsyncOpenAI

from src.core.exceptions import (
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMContextLengthError,
    LLMError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from src.llm._instrumentation import observe_llm_call
from src.llm._openai_format import (
    StreamingToolCallAggregator,
    build_request_kwargs,
    map_finish_reason,
    parse_completion,
    parse_usage,
    strip_model_prefix,
)
from src.llm.base import CompletionChunk, CompletionRequest, CompletionResult

_PROVIDER = "openai"


def _translate_error(exc: Exception) -> LLMError:
    """把 openai SDK 异常翻译为平台异常（不抛出，返回实例）。"""
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError("openai request timed out", cause=exc)
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError("openai rate limited", cause=exc)
    if isinstance(exc, openai.AuthenticationError | openai.PermissionDeniedError):
        return LLMAuthenticationError("openai credentials rejected", cause=exc)
    if isinstance(exc, openai.BadRequestError):
        text = str(exc).lower()
        if "context length" in text or "context_length" in text or "maximum context" in text:
            return LLMContextLengthError("request exceeds model context window", cause=exc)
        if "content_filter" in text or "content management policy" in text:
            return LLMContentFilterError("request blocked by content policy", cause=exc)
        return LLMError(f"openai rejected request: {exc}", cause=exc)
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return LLMServerError(f"openai server error ({exc.status_code})", cause=exc)
    if isinstance(exc, openai.APIConnectionError):
        return LLMServerError("openai connection error", cause=exc)
    return LLMError(f"openai call failed: {exc}", cause=exc)


class OpenAIProvider:
    """:class:`~src.llm.base.LLMProvider` 的 OpenAI 实现。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """初始化 Provider。

        Args:
            api_key: OpenAI API Key。
            base_url: 自定义端点（代理 / Azure 兼容网关）。
            client: 注入现成客户端（测试用）；提供时忽略前两个参数。
        """
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        kwargs = build_request_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=False
        ) as observation:
            try:
                response = await self._client.chat.completions.create(
                    **kwargs, timeout=request.timeout_seconds
                )
            except Exception as exc:
                raise _translate_error(exc) from exc
            result = parse_completion(response, requested_model=request.model)
            observation.usage = result.usage
            return result

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。

        工具调用增量在本方法内聚合，仅在解析完成后以完整
        ``ToolCallRequest`` 下发（接口约定）。
        """
        kwargs = build_request_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=True
        ) as observation:
            aggregator = StreamingToolCallAggregator()
            finish_reason = None
            usage = None
            try:
                response_stream = await self._client.chat.completions.create(
                    **kwargs,
                    timeout=request.timeout_seconds,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for raw_chunk in response_stream:
                    if getattr(raw_chunk, "usage", None) is not None:
                        usage = parse_usage(raw_chunk.usage)
                    if not raw_chunk.choices:
                        continue
                    choice = raw_chunk.choices[0]
                    delta = choice.delta
                    if delta is not None:
                        aggregator.feed(getattr(delta, "tool_calls", None))
                        if delta.content:
                            yield CompletionChunk(content_delta=delta.content)
                    if choice.finish_reason is not None:
                        finish_reason = map_finish_reason(choice.finish_reason)
            except LLMError:
                raise
            except Exception as exc:
                raise _translate_error(exc) from exc

            for tool_call in aggregator.finalize():
                yield CompletionChunk(tool_call=tool_call)
            final_usage = usage or parse_usage(None)
            observation.usage = final_usage
            yield CompletionChunk(
                finish_reason=finish_reason or map_finish_reason(None),
                usage=final_usage,
            )


class OpenAIEmbeddingProvider:
    """:class:`~src.llm.base.EmbeddingProvider` 的 OpenAI 实现。"""

    _BATCH_SIZE = 512

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimension: int,
        base_url: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """初始化 Embedding Provider。

        Args:
            api_key: OpenAI API Key。
            model: embedding 模型名（不带路由前缀）。
            dimension: 输出维度（须与 pgvector 列一致）。
            base_url: 自定义端点。
            client: 注入现成客户端（测试用）。
        """
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = strip_model_prefix(model)
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        """输出向量维度。"""
        return self._dimension

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化。见 :meth:`src.llm.base.EmbeddingProvider.embed`。"""
        results: list[list[float]] = []
        for start in range(0, len(texts), self._BATCH_SIZE):
            batch = list(texts[start : start + self._BATCH_SIZE])
            try:
                response = await self._client.embeddings.create(
                    model=self._model, input=batch, dimensions=self._dimension
                )
            except Exception as exc:
                raise _translate_error(exc) from exc
            ordered = sorted(response.data, key=lambda item: item.index)
            results.extend([item.embedding for item in ordered])
        return results
