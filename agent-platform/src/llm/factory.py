"""LLM 调用栈装配：配置 → Provider → 弹性装饰器 → 路由器。

仅组合根（API deps / worker 入口）调用。
"""

from __future__ import annotations

from src.core.config import LLMSettings
from src.core.exceptions import ConfigurationError
from src.llm.anthropic_provider import AnthropicProvider
from src.llm.base import EmbeddingProvider, LLMProvider
from src.llm.litellm_provider import LiteLLMProvider
from src.llm.openai_provider import OpenAIEmbeddingProvider, OpenAIProvider
from src.llm.resilience import CircuitBreaker, CircuitBreakerProvider, RetryingProvider
from src.llm.router import ProviderRouter


def _wrap_with_resilience(
    provider: LLMProvider, settings: LLMSettings, *, name: str
) -> LLMProvider:
    """套上重试与熔断（熔断在外层，见 resilience 模块文档）。"""
    retrying = RetryingProvider(provider, settings.retry, provider_name=name)
    breaker = CircuitBreaker(name, settings.circuit_breaker)
    return CircuitBreakerProvider(retrying, breaker)


def build_llm_router(settings: LLMSettings) -> ProviderRouter:
    """按配置装配带弹性策略的 Provider 路由器。

    Args:
        settings: LLM 配置；至少需配置一个上游凭证。

    Returns:
        可直接注入上层的路由器。

    Raises:
        ConfigurationError: 没有任何上游被配置。
    """
    providers: dict[str, LLMProvider] = {}

    if settings.openai_api_key is not None:
        providers["openai"] = _wrap_with_resilience(
            OpenAIProvider(
                api_key=settings.openai_api_key.get_secret_value(),
                base_url=settings.openai_base_url,
            ),
            settings,
            name="openai",
        )

    if settings.anthropic_api_key is not None:
        providers["anthropic"] = _wrap_with_resilience(
            AnthropicProvider(api_key=settings.anthropic_api_key.get_secret_value()),
            settings,
            name="anthropic",
        )

    if settings.vllm_base_url is not None:
        providers["vllm"] = _wrap_with_resilience(
            LiteLLMProvider(
                api_base=settings.vllm_base_url,
                api_key=(
                    settings.vllm_api_key.get_secret_value()
                    if settings.vllm_api_key
                    else None
                ),
            ),
            settings,
            name="vllm",
        )

    if not providers:
        raise ConfigurationError(
            "no LLM provider configured; set at least one of "
            "LLM__OPENAI_API_KEY / LLM__ANTHROPIC_API_KEY / LLM__VLLM_BASE_URL"
        )
    return ProviderRouter(providers)


def build_embedding_provider(settings: LLMSettings) -> EmbeddingProvider:
    """装配 Embedding Provider（当前实现：OpenAI embeddings）。

    Raises:
        ConfigurationError: 未配置 OpenAI 凭证。
    """
    if settings.openai_api_key is None:
        raise ConfigurationError(
            "embedding provider requires LLM__OPENAI_API_KEY"
        )
    return OpenAIEmbeddingProvider(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        base_url=settings.openai_base_url,
    )
