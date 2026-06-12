"""重试与熔断装饰器测试。"""

from __future__ import annotations

import pytest
from src.core.config import CircuitBreakerSettings, RetrySettings
from src.core.exceptions import (
    CircuitOpenError,
    LLMAuthenticationError,
    LLMServerError,
    LLMTimeoutError,
)
from src.llm.base import LLMProvider
from src.llm.resilience import (
    CircuitBreaker,
    CircuitBreakerProvider,
    CircuitState,
    RetryingProvider,
)

from tests.llm.fakes import (
    MidStreamFailingProvider,
    ScriptedProvider,
    make_request,
    make_result,
)


def make_retrying(
    inner: LLMProvider, settings: RetrySettings, sleeps: list[float]
) -> RetryingProvider:
    async def record_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return RetryingProvider(
        inner, settings, provider_name="test", sleep=record_sleep, rng=lambda: 1.0
    )


class TestRetryingProvider:
    async def test_succeeds_after_transient_failures(
        self, retry_settings: RetrySettings
    ) -> None:
        inner = ScriptedProvider(
            [LLMTimeoutError("t1"), LLMServerError("t2"), make_result("done")]
        )
        sleeps: list[float] = []
        provider = make_retrying(inner, retry_settings, sleeps)

        result = await provider.complete(make_request())

        assert result.message.content == "done"
        assert inner.calls == 3
        # 指数退避：0.1*2^0=0.1，0.1*2^1=0.2（rng 固定 1.0）
        assert sleeps == [pytest.approx(0.1), pytest.approx(0.2)]

    async def test_backoff_capped_at_max_delay(self) -> None:
        settings = RetrySettings(max_attempts=5, base_delay_seconds=1.0, max_delay_seconds=2.0)
        inner = ScriptedProvider(
            [LLMTimeoutError("e")] * 4 + [make_result("done")]  # type: ignore[list-item]
        )
        sleeps: list[float] = []
        provider = make_retrying(inner, settings, sleeps)

        await provider.complete(make_request())

        assert sleeps == [
            pytest.approx(1.0),
            pytest.approx(2.0),
            pytest.approx(2.0),
            pytest.approx(2.0),
        ]

    async def test_exhausts_attempts(self, retry_settings: RetrySettings) -> None:
        inner = ScriptedProvider([LLMTimeoutError(f"e{i}") for i in range(3)])
        provider = make_retrying(inner, retry_settings, [])

        with pytest.raises(LLMTimeoutError):
            await provider.complete(make_request())
        assert inner.calls == 3

    async def test_non_retryable_fails_immediately(
        self, retry_settings: RetrySettings
    ) -> None:
        inner = ScriptedProvider([LLMAuthenticationError("bad key")])
        provider = make_retrying(inner, retry_settings, [])

        with pytest.raises(LLMAuthenticationError):
            await provider.complete(make_request())
        assert inner.calls == 1

    async def test_stream_retries_before_first_chunk(
        self, retry_settings: RetrySettings
    ) -> None:
        inner = ScriptedProvider([LLMTimeoutError("cold"), make_result("streamed")])
        provider = make_retrying(inner, retry_settings, [])

        chunks = [chunk async for chunk in provider.stream(make_request())]

        assert chunks[0].content_delta == "streamed"
        assert chunks[-1].finish_reason is not None
        assert inner.calls == 2

    async def test_stream_does_not_retry_mid_stream(
        self, retry_settings: RetrySettings
    ) -> None:
        inner = MidStreamFailingProvider(LLMTimeoutError("mid-stream"))
        provider = make_retrying(inner, retry_settings, [])

        received: list[str] = []
        with pytest.raises(LLMTimeoutError):
            async for chunk in provider.stream(make_request()):
                received.append(chunk.content_delta)

        assert received == ["partial"]
        assert inner.calls == 1  # 不二次拉流


class TestCircuitBreaker:
    def test_opens_after_threshold(self, breaker_settings: CircuitBreakerSettings) -> None:
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: 0.0)
        for _ in range(3):
            breaker.acquire()
            breaker.record_failure()

        assert breaker.state is CircuitState.OPEN
        with pytest.raises(CircuitOpenError):
            breaker.acquire()

    def test_success_resets_failure_count(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: 0.0)
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_success()
        breaker.record_failure()
        breaker.record_failure()
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_after_recovery_window(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        now = [0.0]
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: now[0])
        for _ in range(3):
            breaker.record_failure()
        assert breaker.state is CircuitState.OPEN

        now[0] = 10.1  # 超过 recovery_seconds=10
        assert breaker.state is CircuitState.HALF_OPEN
        breaker.acquire()  # 第一个探测放行
        with pytest.raises(CircuitOpenError):
            breaker.acquire()  # 探测槽位已满

    def test_half_open_success_closes(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        now = [0.0]
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: now[0])
        for _ in range(3):
            breaker.record_failure()
        now[0] = 11.0
        breaker.acquire()
        breaker.record_success()
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_failure_reopens(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        now = [0.0]
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: now[0])
        for _ in range(3):
            breaker.record_failure()
        now[0] = 11.0
        breaker.acquire()
        breaker.record_failure()
        assert breaker.state is CircuitState.OPEN
        now[0] = 15.0  # 重新计时，仍在窗口内
        with pytest.raises(CircuitOpenError):
            breaker.acquire()


class TestCircuitBreakerProvider:
    async def test_failures_open_then_fast_fail(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        inner = ScriptedProvider([LLMServerError(f"e{i}") for i in range(3)])
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: 0.0)
        provider = CircuitBreakerProvider(inner, breaker)

        for _ in range(3):
            with pytest.raises(LLMServerError):
                await provider.complete(make_request())

        with pytest.raises(CircuitOpenError):
            await provider.complete(make_request())
        assert inner.calls == 3  # 第 4 次没有到达上游

    async def test_non_retryable_errors_do_not_trip(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        inner = ScriptedProvider(
            [LLMAuthenticationError(f"e{i}") for i in range(5)] + [make_result("ok")]  # type: ignore[list-item]
        )
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: 0.0)
        provider = CircuitBreakerProvider(inner, breaker)

        for _ in range(5):
            with pytest.raises(LLMAuthenticationError):
                await provider.complete(make_request())

        result = await provider.complete(make_request())  # 未熔断
        assert result.message.content == "ok"

    async def test_mid_stream_failure_counts(
        self, breaker_settings: CircuitBreakerSettings
    ) -> None:
        breaker = CircuitBreaker("test", breaker_settings, clock=lambda: 0.0)
        for _ in range(3):
            inner = MidStreamFailingProvider(LLMTimeoutError("mid"))
            provider = CircuitBreakerProvider(inner, breaker)
            with pytest.raises(LLMTimeoutError):
                async for _chunk in provider.stream(make_request()):
                    pass
        assert breaker.state is CircuitState.OPEN
