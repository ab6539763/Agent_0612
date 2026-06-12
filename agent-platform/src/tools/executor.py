"""统一工具执行入口。

职责（见 :class:`~src.tools.base.ToolExecutor` 契约）：
参数校验、超时控制、结果截断、输出脱敏、审计日志、OTel span 与指标。

错误语义：可预期失败（参数非法 / 业务失败 / 超时）以 ``success=False``
的 :class:`~src.tools.base.ToolResult` 返回并回喂模型自我修正；
只有"工具不存在 / 无权限"才抛异常终止。
"""

from __future__ import annotations

import asyncio
import time

from pydantic import ValidationError as PydanticValidationError

from src.core.exceptions import ToolError
from src.core.logging import get_logger
from src.core.observability import (
    TOOL_EXECUTION_SECONDS,
    TOOL_EXECUTIONS_TOTAL,
    get_tracer,
)
from src.core.redaction import redact_text
from src.tools.base import ToolCallInvocation, ToolContext, ToolRegistry, ToolResult

_logger = get_logger(__name__)
_tracer = get_tracer(__name__)

_TRUNCATION_NOTICE = "\n…[output truncated]"


class DefaultToolExecutor:
    """:class:`~src.tools.base.ToolExecutor` 的默认实现。"""

    def __init__(self, registry: ToolRegistry, *, max_result_chars: int = 8_000) -> None:
        """初始化执行器。

        Args:
            registry: 工具注册中心（负责可见性与权限过滤）。
            max_result_chars: 工具输出注入模型上下文的最大字符数。
        """
        self._registry = registry
        self._max_result_chars = max_result_chars

    def _failure(
        self, call: ToolCallInvocation, message: str, outcome: str
    ) -> ToolResult:
        TOOL_EXECUTIONS_TOTAL.labels(tool=call.tool_name, outcome=outcome).inc()
        _logger.warning(
            "tool_execution_failed",
            tool=call.tool_name,
            tool_call_id=call.tool_call_id,
            outcome=outcome,
            error=message,
        )
        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            success=False,
            content=message,
        )

    async def execute(
        self, call: ToolCallInvocation, context: ToolContext
    ) -> ToolResult:
        """执行一次工具调用。见 :meth:`src.tools.base.ToolExecutor.execute`。"""
        tool = self._registry.get(call.tool_name, context.principal)

        try:
            args = tool.args_schema.model_validate(call.raw_arguments)
        except PydanticValidationError as exc:
            errors = "; ".join(
                f"{'.'.join(str(loc) for loc in err['loc'])}: {err['msg']}"
                for err in exc.errors()
            )
            return self._failure(
                call, f"invalid arguments: {errors}", outcome="invalid_arguments"
            )

        start = time.perf_counter()
        with _tracer.start_as_current_span(
            "tool.execute",
            attributes={
                "tool.name": call.tool_name,
                "tool.call_id": call.tool_call_id,
                "run.id": str(context.run_id),
            },
        ):
            try:
                async with asyncio.timeout(context.timeout_seconds):
                    output = await tool.run(args, context)
            except TimeoutError:
                return self._failure(
                    call,
                    f"tool timed out after {context.timeout_seconds}s",
                    outcome="timeout",
                )
            except ToolError as exc:
                return self._failure(call, f"tool failed: {exc.message}", outcome="error")
            except Exception as exc:
                return self._failure(
                    call, f"tool failed unexpectedly: {exc}", outcome="error"
                )
            finally:
                TOOL_EXECUTION_SECONDS.labels(tool=call.tool_name).observe(
                    time.perf_counter() - start
                )

        content = redact_text(output)
        if len(content) > self._max_result_chars:
            content = content[: self._max_result_chars] + _TRUNCATION_NOTICE

        TOOL_EXECUTIONS_TOTAL.labels(tool=call.tool_name, outcome="success").inc()
        _logger.info(
            "tool_executed",
            tool=call.tool_name,
            tool_call_id=call.tool_call_id,
            duration_ms=round((time.perf_counter() - start) * 1000),
            output_chars=len(content),
        )
        return ToolResult(
            tool_call_id=call.tool_call_id,
            tool_name=call.tool_name,
            success=True,
            content=content,
        )
