"""agents 测试替身：脚本化 LLM、内存事件流、内存审批网关。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar
from uuid import UUID, uuid4

from pydantic import BaseModel
from src.agents.approval_gate import _deterministic_id
from src.agents.base import AgentKind, AgentRunRequest, ApprovalRequest, GuardrailConfig
from src.core.exceptions import AgentPlatformError, ConflictError, NotFoundError
from src.core.types import Principal, PrincipalType, TokenUsage, ToolPermission
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ToolCallRequest,
)
from src.tools.base import BaseTool, ToolContext


def make_principal(**overrides: Any) -> Principal:
    """构造具备 read 权限的用户主体。"""
    defaults: dict[str, Any] = {
        "id": "user-1",
        "type": PrincipalType.USER,
        "tenant_id": uuid4(),
        "permissions": frozenset({ToolPermission.READ}),
    }
    defaults.update(overrides)
    return Principal(**defaults)


def make_run_request(
    principal: Principal,
    *,
    agent: AgentKind = AgentKind.REACT,
    input_text: str = "help me",
    guardrails: GuardrailConfig | None = None,
) -> AgentRunRequest:
    """构造最小运行请求。"""
    return AgentRunRequest(
        run_id=uuid4(),
        conversation_id=uuid4(),
        principal=principal,
        agent=agent,
        input=input_text,
        model="openai:gpt-4o",
        guardrails=guardrails or GuardrailConfig(),
    )


def answer(content: str) -> CompletionResult:
    """纯文本回答结果。"""
    return CompletionResult(
        message=ChatMessage(role=ChatRole.ASSISTANT, content=content),
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        model="openai:gpt-4o",
    )


def tool_call_response(
    *calls: ToolCallRequest, reasoning: str = ""
) -> CompletionResult:
    """发起工具调用的结果。"""
    return CompletionResult(
        message=ChatMessage(
            role=ChatRole.ASSISTANT, content=reasoning, tool_calls=tuple(calls)
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5),
        model="openai:gpt-4o",
    )


class ScriptedLLM:
    """按脚本依次出结果的 LLM；stream 用独立脚本。"""

    def __init__(
        self,
        completions: list[CompletionResult | AgentPlatformError] | None = None,
        streams: list[list[CompletionChunk]] | None = None,
    ) -> None:
        self._completions = list(completions or [])
        self._streams = list(streams or [])
        self.requests: list[CompletionRequest] = []
        self.gate_event: asyncio.Event | None = None
        """非 None 时每次 complete 先等待该事件（取消测试用）。"""

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        if self.gate_event is not None:
            await self.gate_event.wait()
        self.requests.append(request)
        if not self._completions:
            raise AssertionError("ScriptedLLM completions exhausted")
        item = self._completions.pop(0)
        if isinstance(item, AgentPlatformError):
            raise item
        return item

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        self.requests.append(request)
        if not self._streams:
            raise AssertionError("ScriptedLLM streams exhausted")
        for chunk in self._streams.pop(0):
            yield chunk


class InMemoryEventStream:
    """EventStream 的内存实现（验证跨进程发布路径）。"""

    def __init__(self) -> None:
        self.published: dict[UUID, list[dict[str, Any]]] = {}

    async def publish(self, run_id: UUID, *, seq: int, payload: dict[str, Any]) -> None:
        self.published.setdefault(run_id, []).append(payload)

    async def last_seq(self, run_id: UUID) -> int:
        events = self.published.get(run_id)
        return events[-1]["seq"] if events else -1

    async def subscribe(
        self, run_id: UUID, *, after_seq: int = -1
    ) -> AsyncIterator[dict[str, Any]]:
        for payload in self.published.get(run_id, []):
            if payload["seq"] > after_seq:
                yield payload


class InMemoryApprovalGate:
    """ApprovalGate 的内存实现（与 SqlApprovalGate 同样的幂等语义）。"""

    def __init__(self) -> None:
        self._approvals: dict[UUID, ApprovalRequest] = {}

    async def create(
        self,
        *,
        run_id: UUID,
        tenant_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        ttl_seconds: int,
    ) -> ApprovalRequest:
        approval_id = _deterministic_id(run_id, arguments)
        if approval_id in self._approvals:
            return self._approvals[approval_id]
        approval = ApprovalRequest(
            id=approval_id,
            run_id=run_id,
            tenant_id=tenant_id,
            tool_name=tool_name,
            arguments=arguments,
            requested_at=datetime.now(tz=UTC),
            expires_at=datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds),
            status="pending",
        )
        self._approvals[approval_id] = approval
        return approval

    async def resolve(
        self, approval_id: UUID, *, approved: bool, resolver: Principal
    ) -> ApprovalRequest:
        approval = self._approvals.get(approval_id)
        if approval is None or approval.tenant_id != resolver.tenant_id:
            raise NotFoundError("approval not found")
        if approval.status != "pending":
            raise ConflictError(f"approval already {approval.status}")
        resolved = approval.model_copy(
            update={
                "status": "approved" if approved else "rejected",
                "resolver": resolver.id,
                "resolved_at": datetime.now(tz=UTC),
            }
        )
        self._approvals[approval_id] = resolved
        return resolved

    async def get(self, approval_id: UUID, *, tenant_id: UUID) -> ApprovalRequest:
        approval = self._approvals.get(approval_id)
        if approval is None or approval.tenant_id != tenant_id:
            raise NotFoundError("approval not found")
        return approval

    async def load(self, approval_id: UUID) -> ApprovalRequest:
        approval = self._approvals.get(approval_id)
        if approval is None:
            raise NotFoundError("approval not found")
        return approval

    async def expire_overdue(self) -> int:
        return 0


class DangerArgs(BaseModel):
    """高危工具参数。"""

    target: str


class DangerousTool(BaseTool):
    """需要人工审批的高危工具（测试用）。"""

    name: ClassVar[str] = "delete_everything"
    description: ClassVar[str] = "dangerous destructive action"
    args_schema: ClassVar[type[BaseModel]] = DangerArgs
    required_permission: ClassVar[ToolPermission] = ToolPermission.READ
    requires_approval: ClassVar[bool] = True

    def __init__(self) -> None:
        self.executions: list[str] = []

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        assert isinstance(args, DangerArgs)
        self.executions.append(args.target)
        return f"deleted {args.target}"
