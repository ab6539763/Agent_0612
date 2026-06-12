"""可观测性：OpenTelemetry 追踪 + Prometheus 指标。

- :func:`configure_tracing` 在进程启动时调用一次；未配置 OTLP 端点时使用
  无导出的本地 TracerProvider（span 上下文仍可用于日志关联）。
- Prometheus 指标为模块级单例；FastAPI / SQLAlchemy / Redis 的自动埋点
  在 API 层启动时接线（阶段 5）。
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
from prometheus_client import Counter, Gauge, Histogram

from src.core.config import Settings

_configured = False


def configure_tracing(settings: Settings) -> None:
    """初始化全局 TracerProvider（幂等）。

    Args:
        settings: 平台配置；``observability.otlp_endpoint`` 非空时启用 OTLP 导出。
    """
    global _configured
    if _configured:
        return

    resource = Resource.create(
        {
            "service.name": settings.observability.service_name,
            "deployment.environment": settings.environment.value,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=TraceIdRatioBased(settings.observability.trace_sample_ratio),
    )
    if settings.observability.otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(endpoint=settings.observability.otlp_endpoint)
            )
        )
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer(name: str) -> trace.Tracer:
    """获取 tracer（通常传 ``__name__``）。"""
    return trace.get_tracer(name)


# ---------------------------------------------------------------------------
# Prometheus 指标（命名遵循 prometheus 惯例：<域>_<对象>_<单位>）
# ---------------------------------------------------------------------------

LLM_REQUESTS_TOTAL = Counter(
    "llm_requests_total",
    "LLM 调用次数",
    labelnames=("provider", "model", "outcome"),
)

LLM_REQUEST_SECONDS = Histogram(
    "llm_request_duration_seconds",
    "LLM 调用时延（流式为总时长）",
    labelnames=("provider", "model"),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0),
)

LLM_TOKENS_TOTAL = Counter(
    "llm_tokens_total",
    "LLM token 消耗",
    labelnames=("provider", "model", "kind"),  # kind: prompt | completion
)

LLM_RETRIES_TOTAL = Counter(
    "llm_retries_total",
    "LLM 重试次数",
    labelnames=("provider",),
)

CIRCUIT_BREAKER_STATE = Gauge(
    "llm_circuit_breaker_state",
    "熔断器状态（0=closed, 1=half_open, 2=open）",
    labelnames=("provider",),
)

TOOL_EXECUTIONS_TOTAL = Counter(
    "tool_executions_total",
    "工具执行次数",
    labelnames=("tool", "outcome"),
)

TOOL_EXECUTION_SECONDS = Histogram(
    "tool_execution_duration_seconds",
    "工具执行时延",
    labelnames=("tool",),
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 15.0, 30.0),
)

AGENT_RUNS_TOTAL = Counter(
    "agent_runs_total",
    "Agent 运行次数",
    labelnames=("agent", "status"),
)

QUEUE_TASKS_TOTAL = Counter(
    "queue_tasks_total",
    "队列任务投递次数",
    labelnames=("task", "outcome"),  # outcome: enqueued | deduplicated | error
)

RETRIEVAL_SECONDS = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG 检索时延（含融合，不含重排）",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
