"""LLM 调用的追踪与指标埋点（Provider 内部共用）。"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from opentelemetry import trace

from src.core.exceptions import AgentPlatformError
from src.core.observability import (
    LLM_REQUEST_SECONDS,
    LLM_REQUESTS_TOTAL,
    LLM_TOKENS_TOTAL,
    get_tracer,
)
from src.core.types import TokenUsage

_tracer = get_tracer(__name__)


class LLMCallObservation:
    """单次调用的观测上下文：实现方在成功路径上回填 usage。"""

    def __init__(self) -> None:
        """初始化（usage 默认为零值）。"""
        self.usage = TokenUsage()


@asynccontextmanager
async def observe_llm_call(
    *, provider: str, model: str, streaming: bool
) -> AsyncIterator[LLMCallObservation]:
    """包裹一次 LLM 调用：OTel span + Prometheus 时延/次数/token 指标。

    Args:
        provider: Provider 名（'openai' / 'anthropic' / 'litellm'）。
        model: 平台模型名（含路由前缀）。
        streaming: 是否流式（流式时观测窗口覆盖整个生成器生命周期）。

    Yields:
        观测上下文，调用方成功后写入 ``usage``。
    """
    start = time.perf_counter()
    with _tracer.start_as_current_span(
        "llm.stream" if streaming else "llm.complete",
        attributes={
            "gen_ai.system": provider,
            "gen_ai.request.model": model,
        },
    ) as span:
        observation = LLMCallObservation()
        try:
            yield observation
        except AgentPlatformError as exc:
            span.set_status(trace.StatusCode.ERROR, exc.code)
            span.record_exception(exc)
            LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, outcome=exc.code).inc()
            raise
        else:
            usage = observation.usage
            span.set_attribute("gen_ai.usage.input_tokens", usage.prompt_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", usage.completion_tokens)
            LLM_REQUESTS_TOTAL.labels(provider=provider, model=model, outcome="success").inc()
            LLM_TOKENS_TOTAL.labels(provider=provider, model=model, kind="prompt").inc(
                usage.prompt_tokens
            )
            LLM_TOKENS_TOTAL.labels(provider=provider, model=model, kind="completion").inc(
                usage.completion_tokens
            )
        finally:
            LLM_REQUEST_SECONDS.labels(provider=provider, model=model).observe(
                time.perf_counter() - start
            )
