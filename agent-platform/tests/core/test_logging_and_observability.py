"""logging 与 observability 配置测试。"""

from __future__ import annotations

import logging

import pytest
from src.core.config import Settings
from src.core.logging import _redaction_processor, configure_logging, get_logger
from src.core.observability import configure_tracing, get_tracer


class TestLogging:
    def test_configure_json_logging(
        self, settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(settings)
        get_logger("test").info("hello", run_id="r-1")
        out = capsys.readouterr().out
        assert '"event": "hello"' in out
        assert '"run_id": "r-1"' in out

    def test_configure_console_logging(self) -> None:
        settings = Settings(_env_file=None)
        settings.observability.json_logs = False
        configure_logging(settings)
        get_logger("test").info("console mode")

    def test_root_handler_installed(self, settings: Settings) -> None:
        configure_logging(settings)
        assert logging.getLogger().handlers

    def test_redaction_processor_masks_event(self) -> None:
        event = _redaction_processor(
            None,
            "info",
            {
                "event": "llm_call",
                "api_key": "sk-secret-123",
                "detail": {"authorization": "Bearer abcdef123456789"},
                "user_email": "alice@example.com",
            },
        )
        assert event["api_key"] == "[REDACTED]"
        assert event["detail"]["authorization"] == "[REDACTED]"
        assert "alice@" not in event["user_email"]


class TestObservability:
    def test_configure_tracing_without_otlp(self, settings: Settings) -> None:
        configure_tracing(settings)
        tracer = get_tracer("test")
        with tracer.start_as_current_span("unit-test-span") as span:
            assert span.get_span_context().trace_id != 0

    def test_configure_idempotent(self, settings: Settings) -> None:
        configure_tracing(settings)
        configure_tracing(settings)  # 第二次为空操作，不抛错
