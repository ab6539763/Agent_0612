"""litellm 兜底适配器：覆盖 vLLM 及其他 OpenAI-compatible 长尾模型（ADR-0003）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import litellm

from src.core.exceptions import (
    LLMAuthenticationError,
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
)
from src.llm.base import CompletionChunk, CompletionRequest, CompletionResult

_PROVIDER = "litellm"


def _translate_error(exc: Exception) -> LLMError:
    """把 litellm 异常翻译为平台异常（litellm 异常镜像 openai 异常体系）。"""
    if isinstance(exc, litellm.exceptions.Timeout):
        return LLMTimeoutError("litellm request timed out", cause=exc)
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return LLMRateLimitError("upstream rate limited", cause=exc)
    if isinstance(
        exc,
        litellm.exceptions.AuthenticationError | litellm.exceptions.PermissionDeniedError,
    ):
        return LLMAuthenticationError("upstream credentials rejected", cause=exc)
    if isinstance(exc, litellm.exceptions.ContextWindowExceededError):
        return LLMContextLengthError("request exceeds model context window", cause=exc)
    if isinstance(
        exc,
        litellm.exceptions.ServiceUnavailableError
        | litellm.exceptions.InternalServerError
        | litellm.exceptions.APIConnectionError,
    ):
        return LLMServerError("upstream server error", cause=exc)
    return LLMError(f"litellm call failed: {exc}", cause=exc)


class LiteLLMProvider:
    """:class:`~src.llm.base.LLMProvider` 的 litellm 实现。

    模型名映射：``vllm:<name>`` → ``hosted_vllm/<name>``（配合 ``api_base``）；
    其他前缀剥离后原样传给 litellm（由其自身的 provider 前缀机制处理）。
    """

    def __init__(
        self, *, api_base: str | None = None, api_key: str | None = None
    ) -> None:
        """初始化 Provider。

        Args:
            api_base: OpenAI-compatible 端点（vLLM 服务地址）。
            api_key: 端点鉴权 key（vLLM 通常不需要）。
        """
        self._api_base = api_base
        self._api_key = api_key

    def _build_kwargs(self, request: CompletionRequest) -> dict[str, Any]:
        kwargs = build_request_kwargs(request)
        prefix, _, rest = request.model.partition(":")
        if prefix == "vllm" and rest:
            kwargs["model"] = f"hosted_vllm/{rest}"
        if self._api_base:
            kwargs["api_base"] = self._api_base
        if self._api_key:
            kwargs["api_key"] = self._api_key
        kwargs["timeout"] = request.timeout_seconds
        return kwargs

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        kwargs = self._build_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=False
        ) as observation:
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                raise _translate_error(exc) from exc
            result = parse_completion(response, requested_model=request.model)
            observation.usage = result.usage
            return result

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。"""
        kwargs = self._build_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=True
        ) as observation:
            aggregator = StreamingToolCallAggregator()
            finish_reason = None
            usage = None
            try:
                response_stream = await litellm.acompletion(
                    **kwargs, stream=True, stream_options={"include_usage": True}
                )
                async for raw_chunk in response_stream:
                    if getattr(raw_chunk, "usage", None) is not None:
                        usage = parse_usage(raw_chunk.usage)
                    choices = getattr(raw_chunk, "choices", None)
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.delta
                    if delta is not None:
                        aggregator.feed(getattr(delta, "tool_calls", None))
                        if getattr(delta, "content", None):
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
