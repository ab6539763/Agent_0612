"""Agent 编排层对外接口（ADR-0002）。

本模块定义编排层与外界（API、任务队列）的全部契约：

- :class:`AgentRunRequest` / :class:`AgentState`：运行输入与图状态 Schema。
- :class:`AgentRuntime`：运行入口，产出 :data:`~src.agents.events.AgentEvent` 流。
- :class:`ApprovalGate`：Human-in-the-loop 审批的持久化接口。

LangGraph 类型（StateGraph、Command 等）不出现在本文件——它们是阶段 3
实现细节，被封装在具体 Runtime 实现内。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.agents.events import AgentEvent, PlanStep
from src.core.types import Principal, TokenUsage
from src.llm.base import ChatMessage


class AgentKind(StrEnum):
    """内置图模板。"""

    REACT = "react"
    """ReAct 推理循环：reason → act → observe 条件环。"""

    PLANNER_EXECUTOR = "planner_executor"
    """Planner-Executor：plan → route → executor 子图 → review → (replan|respond)。"""


class GuardrailConfig(BaseModel):
    """运行护栏配置。"""

    model_config = ConfigDict(frozen=True)

    max_iterations: int = Field(default=15, gt=0, le=100)
    token_budget: int = Field(
        default=200_000, gt=0, description="单次运行累计 token 上限。"
    )
    run_timeout_seconds: float = Field(default=600.0, gt=0)


class AgentRunRequest(BaseModel):
    """一次 Agent 运行的全部输入。"""

    model_config = ConfigDict(frozen=True)

    run_id: UUID
    conversation_id: UUID
    principal: Principal
    agent: AgentKind = AgentKind.REACT
    input: str = Field(min_length=1, description="用户输入（已通过注入检测）。")
    model: str = Field(description="带路由前缀的模型名，如 'openai:gpt-4o'。")
    guardrails: GuardrailConfig = GuardrailConfig()
    emit_reasoning: bool = Field(
        default=True, description="是否下发 reasoning_delta 事件。"
    )


class AgentState(BaseModel):
    """图状态 Schema（LangGraph 状态通道的类型声明）。

    约定：必须可 JSON 序列化（checkpoint 要求）；``messages`` 通道在图定义中
    配置追加 reducer，其余通道为覆盖语义。
    """

    run_id: UUID
    conversation_id: UUID
    messages: list[ChatMessage] = Field(
        default_factory=list,
        description="运行内消息轨迹（追加 reducer 通道）。",
    )
    plan: list[PlanStep] = Field(
        default_factory=list, description="Planner-Executor 模式下的当前计划。"
    )
    iteration: int = Field(default=0, ge=0, description="ReAct 已完成的循环次数。")
    usage: TokenUsage = TokenUsage()
    final_answer: str | None = None
    termination_reason: str | None = Field(
        default=None,
        description="非正常终止原因：'max_iterations' / 'token_budget' / 'cancelled'。",
    )


class RunStatus(StrEnum):
    """运行状态机。

    合法迁移：
    ``queued → running → {waiting_approval, succeeded, failed, cancelled}``；
    ``waiting_approval → {running, cancelled, failed(审批超时/拒绝)}``。
    """

    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@runtime_checkable
class AgentRuntime(Protocol):
    """Agent 运行入口（API 层与 worker 唯一依赖的编排接口）。

    实现要求（阶段 3）：
    - 事件 ``seq`` 单调递增，且同步写入 Redis Stream 供断线续传/跨进程下发。
    - 触发审批中断时：持久化审批请求、发出 ``approval_required`` 事件、
      正常结束本次迭代（checkpoint 已由 LangGraph 落盘）。
    - 所有异常翻译为 ``run_failed`` 事件后再抛出/记录，不留挂起的流。
    """

    def run_stream(self, request: AgentRunRequest) -> AsyncIterator[AgentEvent]:
        """启动一次新运行并流式产出事件。

        Args:
            request: 运行输入。

        Yields:
            Agent 事件流，以 ``run_finished`` / ``run_failed`` 终止。

        Raises:
            GuardrailViolationError: 护栏配置非法（运行中触达护栏不抛错，
                以 ``run_failed`` 事件 + termination_reason 表达）。
        """
        ...

    def resume_stream(
        self, run_id: UUID, *, approval_id: UUID
    ) -> AsyncIterator[AgentEvent]:
        """审批通过后从 checkpoint 恢复运行。

        Args:
            run_id: 处于 ``waiting_approval`` 状态的运行。
            approval_id: 已决议的审批请求。

        Yields:
            恢复后的事件流（seq 接续暂停前的序号）。

        Raises:
            NotFoundError: 运行或审批不存在。
            ConflictError: 运行不处于可恢复状态，或审批未决议。
            CheckpointError: checkpoint 恢复失败。
        """
        ...

    async def cancel(self, run_id: UUID, *, principal: Principal) -> None:
        """取消运行（运行中或等待审批均可取消）。

        Raises:
            NotFoundError: 运行不存在或不属于该租户。
            ConflictError: 运行已处于终态。
        """
        ...


# ---------------------------------------------------------------------------
# Human-in-the-loop 审批
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    """一条待审批请求（持久化于 PostgreSQL ``approvals`` 表）。"""

    model_config = ConfigDict(frozen=True)

    id: UUID
    run_id: UUID
    tenant_id: UUID
    tool_name: str
    arguments: dict[str, Any] = Field(description="待执行操作的参数快照。")
    requested_at: datetime
    expires_at: datetime
    status: str = Field(description="'pending' / 'approved' / 'rejected' / 'expired'。")
    resolver: str | None = None
    resolved_at: datetime | None = None


@runtime_checkable
class ApprovalGate(Protocol):
    """审批请求的持久化与决议接口。

    审批 API（``POST /v1/approvals/{id}``）与 AgentRuntime 共同依赖本接口；
    过期处理由定时任务调用 :meth:`expire_overdue`。
    """

    async def create(
        self,
        *,
        run_id: UUID,
        tenant_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        ttl_seconds: int,
    ) -> ApprovalRequest:
        """创建审批请求（运行进入 ``waiting_approval`` 时调用）。"""
        ...

    async def resolve(
        self,
        approval_id: UUID,
        *,
        approved: bool,
        resolver: Principal,
    ) -> ApprovalRequest:
        """决议审批。

        Args:
            approval_id: 审批请求 ID。
            approved: 通过 / 拒绝。
            resolver: 审批人（须具备 'approvals:write' scope，且同租户）。

        Returns:
            决议后的审批请求。

        Raises:
            NotFoundError: 请求不存在或跨租户。
            ConflictError: 请求已决议或已过期。
        """
        ...

    async def get(self, approval_id: UUID, *, tenant_id: UUID) -> ApprovalRequest:
        """查询审批请求。

        Raises:
            NotFoundError: 请求不存在或跨租户。
        """
        ...

    async def expire_overdue(self) -> int:
        """把超时未决议的请求标记为 expired，并使对应运行失败。

        Returns:
            本次处理的过期请求数。
        """
        ...
