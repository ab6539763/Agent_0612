"""structlog 结构化日志配置。

- 生产输出单行 JSON，开发输出彩色控制台。
- 全部事件经脱敏 processor（键名 + 内容两级）。
- ``trace_id`` / ``run_id`` / ``tenant_id`` 等通过 contextvars 绑定，
  由中间件 / runtime 在边界处 ``bind_contextvars``。
- 标准库 logging（uvicorn、sqlalchemy 等）统一接入同一渲染管道。
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from src.core.config import Settings
from src.core.redaction import redact_value


def _redaction_processor(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """对整个事件字典做递归脱敏（structlog processor）。"""
    redacted = redact_value(dict(event_dict))
    assert isinstance(redacted, dict)
    return redacted


def _add_otel_trace_id(
    logger: WrappedLogger, method_name: str, event_dict: EventDict
) -> EventDict:
    """把当前 OTel span 的 trace_id 注入日志（无活跃 span 时跳过）。"""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        ctx = span.get_span_context()
        if ctx.is_valid:
            event_dict.setdefault("trace_id", format(ctx.trace_id, "032x"))
    except ImportError:  # pragma: no cover - otel 是必装依赖，防御性兜底
        pass
    return event_dict


def configure_logging(settings: Settings) -> None:
    """初始化全局日志（进程启动时调用一次，幂等）。

    Args:
        settings: 平台配置（决定级别与输出格式）。
    """
    level = logging.getLevelNamesMapping().get(
        settings.observability.log_level.upper(), logging.INFO
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _add_otel_trace_id,
        _redaction_processor,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Any
    if settings.observability.json_logs:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # 标准库 logging（uvicorn / sqlalchemy / arq）接入同一渲染管道
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """获取绑定模块名的 logger。

    Args:
        name: 通常传 ``__name__``。

    Returns:
        structlog BoundLogger。
    """
    return structlog.stdlib.get_logger(name)
