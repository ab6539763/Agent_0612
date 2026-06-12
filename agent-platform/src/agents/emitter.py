"""事件发射器：图节点产生 AgentEvent 的唯一出口。

设计（ADR-0007）：不依赖 LangGraph ``astream_events`` 的内部事件格式，
节点通过 ``config["configurable"]["emitter"]`` 拿到本类实例主动发事件。
事件同时进入进程内队列（本地 SSE 消费）与 Redis Stream（跨进程下发 +
断线续传），``seq`` 由发射器单调分配。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from src.agents.base import ApprovalRequest
from src.agents.events import (
    AgentEvent,
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    MessageCompletedEvent,
    MessageDeltaEvent,
    PlanCreatedEvent,
    PlanStep,
    PlanUpdatedEvent,
    ReasoningDeltaEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from src.core.logging import get_logger
from src.core.redaction import redact_value
from src.core.types import TokenUsage
from src.infrastructure.base import EventStream
from src.tools.base import ToolResult

_logger = get_logger(__name__)

_SENTINEL = None
"""队列终止哨兵：消费方收到后结束迭代。"""


class EventEmitter:
    """单次运行（或恢复段）的事件发射器。"""

    def __init__(
        self,
        run_id: UUID,
        *,
        start_seq: int = 0,
        event_stream: EventStream | None = None,
        emit_reasoning: bool = True,
    ) -> None:
        """初始化发射器。

        Args:
            run_id: 运行 ID。
            start_seq: 起始序号（恢复时续接 Redis Stream 的最后序号 + 1）。
            event_stream: 跨进程事件流；None 时仅进程内队列（测试）。
            emit_reasoning: False 时丢弃 reasoning_delta 事件（租户配置）。
        """
        self._run_id = run_id
        self._next_seq = start_seq
        self._event_stream = event_stream
        self._emit_reasoning = emit_reasoning
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()

    @property
    def queue(self) -> asyncio.Queue[AgentEvent | None]:
        """事件队列（runtime 消费；``None`` 为终止哨兵）。"""
        return self._queue

    async def _emit(self, event: AgentEvent) -> None:
        self._queue.put_nowait(event)
        if self._event_stream is not None:
            payload = event.model_dump(mode="json")
            await self._event_stream.publish(self._run_id, seq=event.seq, payload=payload)

    def _stamp(self) -> dict[str, Any]:
        seq = self._next_seq
        self._next_seq += 1
        return {"run_id": self._run_id, "seq": seq, "timestamp": datetime.now(tz=UTC)}

    async def close(self) -> None:
        """发送终止哨兵（任何终态事件之后调用一次）。"""
        self._queue.put_nowait(_SENTINEL)

    # ------------------------------------------------------------------ #
    # 事件方法（参数中的不可信内容统一在此脱敏）
    # ------------------------------------------------------------------ #

    async def run_started(self, *, agent: str, model: str) -> None:
        """发出 run_started。"""
        await self._emit(RunStartedEvent(agent=agent, model=model, **self._stamp()))

    async def run_finished(self, *, usage: TokenUsage, duration_ms: int) -> None:
        """发出 run_finished（终态）。"""
        await self._emit(
            RunFinishedEvent(usage=usage, duration_ms=duration_ms, **self._stamp())
        )

    async def run_failed(self, *, error: dict[str, Any]) -> None:
        """发出 run_failed（终态），error 为 Problem Details。"""
        await self._emit(RunFailedEvent(error=error, **self._stamp()))

    async def plan_created(self, steps: tuple[PlanStep, ...]) -> None:
        """发出 plan_created。"""
        await self._emit(PlanCreatedEvent(steps=steps, **self._stamp()))

    async def plan_updated(self, steps: tuple[PlanStep, ...]) -> None:
        """发出 plan_updated。"""
        await self._emit(PlanUpdatedEvent(steps=steps, **self._stamp()))

    async def step_started(self, step: str) -> None:
        """发出 step_started。"""
        await self._emit(StepStartedEvent(step=step, **self._stamp()))

    async def step_finished(self, step: str) -> None:
        """发出 step_finished。"""
        await self._emit(StepFinishedEvent(step=step, **self._stamp()))

    async def reasoning_delta(self, delta: str) -> None:
        """发出 reasoning_delta（按配置可丢弃；空文本跳过）。"""
        if not self._emit_reasoning or not delta:
            return
        await self._emit(ReasoningDeltaEvent(delta=delta, **self._stamp()))

    async def tool_call(
        self, *, tool_call_id: str, tool_name: str, arguments: dict[str, Any]
    ) -> None:
        """发出 tool_call（参数脱敏后下发）。"""
        await self._emit(
            ToolCallEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=cast("dict[str, Any]", redact_value(arguments)),
                **self._stamp(),
            )
        )

    async def tool_result(self, result: ToolResult) -> None:
        """发出 tool_result（执行器已截断，此处再脱敏 artifacts）。"""
        await self._emit(
            ToolResultEvent(
                tool_call_id=result.tool_call_id,
                tool_name=result.tool_name,
                success=result.success,
                content=result.content,
                artifacts=cast("dict[str, Any]", redact_value(result.artifacts)),
                **self._stamp(),
            )
        )

    async def message_delta(self, delta: str) -> None:
        """发出 message_delta（空文本跳过）。"""
        if not delta:
            return
        await self._emit(MessageDeltaEvent(delta=delta, **self._stamp()))

    async def message_completed(self, content: str) -> None:
        """发出 message_completed。"""
        await self._emit(MessageCompletedEvent(content=content, **self._stamp()))

    async def approval_required(self, approval: ApprovalRequest) -> None:
        """发出 approval_required（参数快照脱敏）。"""
        await self._emit(
            ApprovalRequiredEvent(
                approval_id=approval.id,
                tool_name=approval.tool_name,
                arguments=cast("dict[str, Any]", redact_value(approval.arguments)),
                expires_at=approval.expires_at,
                **self._stamp(),
            )
        )

    async def approval_resolved(self, approval: ApprovalRequest) -> None:
        """发出 approval_resolved。"""
        await self._emit(
            ApprovalResolvedEvent(
                approval_id=approval.id,
                approved=approval.status == "approved",
                resolver=approval.resolver or "unknown",
                **self._stamp(),
            )
        )
