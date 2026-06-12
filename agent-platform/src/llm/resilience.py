"""LLM 调用弹性装饰器：重试（指数退避 + 全抖动）与熔断（ADR-0003）。

组合顺序（factory 装配）::

    ProviderRouter → CircuitBreakerProvider → RetryingProvider → 具体 Provider

熔断在重试外层：重试耗尽才计为一次熔断失败，避免单请求的多次重试快速打满
失败窗口。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from enum import IntEnum

from src.core.config import CircuitBreakerSettings, RetrySettings
from src.core.exceptions import CircuitOpenError, LLMError
from src.core.logging import get_logger
from src.core.observability import CIRCUIT_BREAKER_STATE, LLM_RETRIES_TOTAL
from src.llm.base import (
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    LLMProvider,
)

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 重试
# ---------------------------------------------------------------------------


class RetryingProvider:
    """重试装饰器：仅对 ``retryable=True`` 的 ``LLMError`` 重试。

    流式语义：只在产出第一个 chunk **之前**重试；一旦内容开始下发，
    中途失败直接上抛（已下发的增量无法安全重放）。
    """

    def __init__(
        self,
        inner: LLMProvider,
        settings: RetrySettings,
        *,
        provider_name: str,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        rng: Callable[[], float] = random.random,
    ) -> None:
        """初始化装饰器。

        Args:
            inner: 被包装的 Provider。
            settings: 重试策略。
            provider_name: 指标与日志中的 Provider 标识。
            sleep: 退避等待函数（测试注入）。
            rng: [0,1) 随机源，全抖动用（测试注入）。
        """
        self._inner = inner
        self._settings = settings
        self._provider_name = provider_name
        self._sleep = sleep
        self._rng = rng

    def _backoff_seconds(self, attempt: int) -> float:
        """第 ``attempt`` 次失败后的等待时长（全抖动：U(0, min(cap, base*2^n))）。"""
        ceiling: float = min(
            self._settings.max_delay_seconds,
            self._settings.base_delay_seconds * (2 ** (attempt - 1)),
        )
        return ceiling * float(self._rng())

    async def _wait_before_retry(self, attempt: int, exc: LLMError) -> None:
        delay = self._backoff_seconds(attempt)
        LLM_RETRIES_TOTAL.labels(provider=self._provider_name).inc()
        _logger.warning(
            "llm_retry",
            provider=self._provider_name,
            attempt=attempt,
            max_attempts=self._settings.max_attempts,
            delay_seconds=round(delay, 3),
            error_code=exc.code,
        )
        await self._sleep(delay)

    def _should_retry(self, exc: LLMError, attempt: int) -> bool:
        return exc.retryable and attempt < self._settings.max_attempts

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """带重试的非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        attempt = 1
        while True:
            try:
                return await self._inner.complete(request)
            except LLMError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(attempt, exc)
                attempt += 1

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """带重试的流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。"""
        attempt = 1
        while True:
            iterator = aiter(self._inner.stream(request))
            try:
                first_chunk = await anext(iterator)
            except StopAsyncIteration:
                return
            except LLMError as exc:
                if not self._should_retry(exc, attempt):
                    raise
                await self._wait_before_retry(attempt, exc)
                attempt += 1
                continue
            yield first_chunk
            async for chunk in iterator:
                yield chunk
            return


# ---------------------------------------------------------------------------
# 熔断
# ---------------------------------------------------------------------------


class CircuitState(IntEnum):
    """熔断器状态（数值即 Prometheus gauge 取值）。"""

    CLOSED = 0
    HALF_OPEN = 1
    OPEN = 2


class CircuitBreaker:
    """连续失败阈值 + 恢复窗口 + 半开探测的熔断器。

    单事件循环内使用，状态变更都发生在 await 点之间，无需加锁。
    """

    def __init__(
        self,
        name: str,
        settings: CircuitBreakerSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """初始化熔断器。

        Args:
            name: 熔断器标识（Provider 名），用于指标与日志。
            settings: 熔断配置。
            clock: 单调时钟（测试注入）。
        """
        self._name = name
        self._settings = settings
        self._clock = clock
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._probes_in_flight = 0
        self._publish_state()

    @property
    def state(self) -> CircuitState:
        """当前状态（开启态超过恢复窗口时惰性迁移到半开）。"""
        if (
            self._state is CircuitState.OPEN
            and self._clock() - self._opened_at >= self._settings.recovery_seconds
        ):
            self._transition(CircuitState.HALF_OPEN)
            self._probes_in_flight = 0
        return self._state

    def _publish_state(self) -> None:
        CIRCUIT_BREAKER_STATE.labels(provider=self._name).set(int(self._state))

    def _transition(self, new_state: CircuitState) -> None:
        if new_state is not self._state:
            _logger.warning(
                "circuit_breaker_transition",
                provider=self._name,
                from_state=self._state.name,
                to_state=new_state.name,
            )
        self._state = new_state
        self._publish_state()

    def acquire(self) -> None:
        """请求放行；半开态占用一个探测槽位。

        Raises:
            CircuitOpenError: 熔断开启，或半开态探测槽位已满。
        """
        state = self.state
        if state is CircuitState.CLOSED:
            return
        if state is CircuitState.HALF_OPEN:
            if self._probes_in_flight < self._settings.half_open_max_probes:
                self._probes_in_flight += 1
                return
            raise CircuitOpenError(
                "circuit breaker half-open, probe slots exhausted",
                details={"provider": self._name},
            )
        raise CircuitOpenError(
            "circuit breaker open",
            details={
                "provider": self._name,
                "retry_after_seconds": max(
                    0.0,
                    self._settings.recovery_seconds - (self._clock() - self._opened_at),
                ),
            },
        )

    def record_success(self) -> None:
        """记录一次成功（半开态成功即恢复闭合）。"""
        if self._state is CircuitState.HALF_OPEN:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)
            self._transition(CircuitState.CLOSED)
        self._consecutive_failures = 0

    def record_failure(self) -> None:
        """记录一次失败（半开态失败立即重新开启）。"""
        if self._state is CircuitState.HALF_OPEN:
            self._probes_in_flight = max(0, self._probes_in_flight - 1)
            self._opened_at = self._clock()
            self._transition(CircuitState.OPEN)
            return
        self._consecutive_failures += 1
        if (
            self._state is CircuitState.CLOSED
            and self._consecutive_failures >= self._settings.failure_threshold
        ):
            self._opened_at = self._clock()
            self._transition(CircuitState.OPEN)


class CircuitBreakerProvider:
    """熔断装饰器。

    失败计数口径：仅 ``retryable=True`` 的 ``LLMError``（超时 / 5xx / 限流）
    计入——鉴权失败、上下文超长等确定性错误不会因为换时间重试而恢复，
    计入只会让熔断掩盖真实根因。
    """

    def __init__(
        self,
        inner: LLMProvider,
        breaker: CircuitBreaker,
    ) -> None:
        """初始化装饰器。

        Args:
            inner: 被包装的 Provider（通常已套 RetryingProvider）。
            breaker: 熔断器实例（每个 Provider 一个）。
        """
        self._inner = inner
        self._breaker = breaker

    def _record(self, exc: LLMError) -> None:
        if exc.retryable and not isinstance(exc, CircuitOpenError):
            self._breaker.record_failure()

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """带熔断的非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        self._breaker.acquire()
        try:
            result = await self._inner.complete(request)
        except LLMError as exc:
            self._record(exc)
            raise
        self._breaker.record_success()
        return result

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """带熔断的流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。

        流中途失败同样计入熔断（上游不稳定的信号与建连失败等价）。
        """
        self._breaker.acquire()
        try:
            async for chunk in self._inner.stream(request):
                yield chunk
        except LLMError as exc:
            self._record(exc)
            raise
        self._breaker.record_success()
