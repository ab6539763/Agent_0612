"""Agent 流式事件协议（ADR-0007）。

所有 Agent 运行的对外输出统一为 :data:`AgentEvent` 判别联合。
LangGraph 原始事件在 ``src/agents`` 内翻译为这些事件，框架细节不出模块边界；
API 层把事件序列化为 SSE（``event:`` = type，``data:`` = JSON，``id:`` = seq）。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import TokenUsage


class BaseEvent(BaseModel):
    """所有事件的公共字段。"""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    seq: int = Field(ge=0, description="run 内单调递增序号，SSE 断线续传锚点。")
    timestamp: datetime


# --- 运行生命周期 -----------------------------------------------------------


class RunStartedEvent(BaseEvent):
    """运行开始。"""

    type: Literal["run_started"] = "run_started"
    agent: str = Field(description="图模板名：'react' / 'planner_executor'。")
    model: str


class RunFinishedEvent(BaseEvent):
    """运行成功结束（流的终态之一，此后服务端关闭 SSE）。"""

    type: Literal["run_finished"] = "run_finished"
    usage: TokenUsage
    duration_ms: int = Field(ge=0)


class RunFailedEvent(BaseEvent):
    """运行失败（流的终态之一）。错误以 Problem Details 形式承载。"""

    type: Literal["run_failed"] = "run_failed"
    error: dict[str, Any] = Field(
        description="RFC 9457 Problem Details（AgentPlatformError.to_problem()）。"
    )


# --- 计划 -------------------------------------------------------------------


class PlanStep(BaseModel):
    """计划中的单个步骤。"""

    model_config = ConfigDict(frozen=True)

    index: int = Field(ge=0)
    description: str
    assignee: str = Field(description="负责执行的子 Agent / executor 名。")
    status: Literal["pending", "running", "done", "failed", "skipped"] = "pending"


class PlanCreatedEvent(BaseEvent):
    """Planner 产出初始计划。"""

    type: Literal["plan_created"] = "plan_created"
    steps: tuple[PlanStep, ...]


class PlanUpdatedEvent(BaseEvent):
    """计划修订（review 节点触发 replan 或步骤状态变更）。"""

    type: Literal["plan_updated"] = "plan_updated"
    steps: tuple[PlanStep, ...]


# --- 节点级进度与推理 ---------------------------------------------------------


class StepStartedEvent(BaseEvent):
    """图节点开始执行。"""

    type: Literal["step_started"] = "step_started"
    step: str = Field(description="节点名：'reason' / 'act' / 'plan' / ...")


class StepFinishedEvent(BaseEvent):
    """图节点执行结束。"""

    type: Literal["step_finished"] = "step_finished"
    step: str


class ReasoningDeltaEvent(BaseEvent):
    """推理过程增量文本（可按租户配置关闭下发）。"""

    type: Literal["reasoning_delta"] = "reasoning_delta"
    delta: str


# --- 工具调用 ----------------------------------------------------------------


class ToolCallEvent(BaseEvent):
    """模型发起工具调用（参数已脱敏）。"""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


class ToolResultEvent(BaseEvent):
    """工具执行结果（内容已脱敏并截断）。"""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    artifacts: dict[str, Any] = Field(default_factory=dict)


# --- 最终回答 ----------------------------------------------------------------


class MessageDeltaEvent(BaseEvent):
    """最终回答的增量 token。"""

    type: Literal["message_delta"] = "message_delta"
    delta: str


class MessageCompletedEvent(BaseEvent):
    """最终回答完成（完整文本，便于客户端校验拼接结果）。"""

    type: Literal["message_completed"] = "message_completed"
    content: str


# --- Human-in-the-loop --------------------------------------------------------


class ApprovalRequiredEvent(BaseEvent):
    """运行暂停等待人工审批（高危工具调用前触发）。"""

    type: Literal["approval_required"] = "approval_required"
    approval_id: UUID
    tool_name: str
    arguments: dict[str, Any] = Field(description="待审批操作的参数快照（已脱敏）。")
    expires_at: datetime = Field(description="审批超时时间，超时按拒绝处理。")


class ApprovalResolvedEvent(BaseEvent):
    """审批已决议，运行恢复或终止。"""

    type: Literal["approval_resolved"] = "approval_resolved"
    approval_id: UUID
    approved: bool
    resolver: str = Field(description="审批人标识。")


# --- 保活 ---------------------------------------------------------------------


class HeartbeatEvent(BaseEvent):
    """SSE 保活（15s 无业务事件时下发）。"""

    type: Literal["heartbeat"] = "heartbeat"


AgentEvent = Annotated[
    RunStartedEvent
    | RunFinishedEvent
    | RunFailedEvent
    | PlanCreatedEvent
    | PlanUpdatedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | ReasoningDeltaEvent
    | ToolCallEvent
    | ToolResultEvent
    | MessageDeltaEvent
    | MessageCompletedEvent
    | ApprovalRequiredEvent
    | ApprovalResolvedEvent
    | HeartbeatEvent,
    Field(discriminator="type"),
]
"""Agent 事件判别联合：以 ``type`` 字段判别，可直接用于 Pydantic 序列化/反序列化。"""
