"""LangGraph Agent 运行时：:class:`~src.agents.base.AgentRuntime` 的实现。

职责：图选择与启动、事件流装配（队列 + Redis Stream）、interrupt 检测与
审批衔接、护栏终止与异常到 ``run_failed`` 事件的翻译、恢复时的 seq 续接。

阶段 3 边界（阶段 5 接线）：运行状态机到 PostgreSQL 的持久化、跨进程取消
（Redis 取消标记）。当前取消与恢复所需的原始请求保存在进程内
:class:`InMemoryRunRequestStore`；生产部署在阶段 5 替换为 PG 实现。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, Command

from src.agents.base import (
    AgentKind,
    AgentRunRequest,
    AgentState,
    ApprovalGate,
)
from src.agents.emitter import EventEmitter
from src.agents.graphs.common import (
    TERMINATION_CANCELLED,
    GraphDeps,
)
from src.agents.graphs.planner import build_planner_graph
from src.agents.graphs.react import build_react_graph
from src.core.exceptions import (
    AgentPlatformError,
    ConflictError,
    GraphExecutionError,
    GuardrailViolationError,
    NotFoundError,
    RunCancelledError,
)
from src.core.logging import get_logger
from src.core.observability import AGENT_RUNS_TOTAL
from src.core.types import Principal, TokenUsage
from src.infrastructure.base import EventStream
from src.llm.base import ChatMessage, ChatRole, LLMProvider
from src.tools.base import ToolExecutor, ToolRegistry

_logger = get_logger(__name__)


@runtime_checkable
class RunRequestStore(Protocol):
    """运行请求的存取接口（恢复 / 取消需要原始请求上下文）。"""

    async def save(self, request: AgentRunRequest) -> None:
        """保存请求。"""
        ...

    async def load(self, run_id: UUID) -> AgentRunRequest:
        """加载请求。

        Raises:
            NotFoundError: 运行未知。
        """
        ...


class InMemoryRunRequestStore:
    """进程内实现（单实例部署 / 测试）。生产多副本场景在阶段 5 换 PG 实现。"""

    def __init__(self) -> None:
        """初始化空存储。"""
        self._requests: dict[UUID, AgentRunRequest] = {}

    async def save(self, request: AgentRunRequest) -> None:
        """保存请求。"""
        self._requests[request.run_id] = request

    async def load(self, run_id: UUID) -> AgentRunRequest:
        """加载请求。

        Raises:
            NotFoundError: 运行未知。
        """
        request = self._requests.get(run_id)
        if request is None:
            raise NotFoundError("run not found", details={"run_id": str(run_id)})
        return request


class LangGraphAgentRuntime:
    """:class:`~src.agents.base.AgentRuntime` 的 LangGraph 实现。"""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        registry: ToolRegistry,
        executor: ToolExecutor,
        gate: ApprovalGate,
        checkpointer: Checkpointer,
        event_stream: EventStream | None = None,
        request_store: RunRequestStore | None = None,
    ) -> None:
        """初始化运行时（组合根装配）。

        Args:
            llm: 模型调用栈（已套弹性装饰器的路由器）。
            registry: 工具注册中心。
            executor: 工具执行器。
            gate: 审批网关。
            checkpointer: LangGraph 状态持久化器。
            event_stream: 跨进程事件流；None 时仅进程内（测试）。
            request_store: 运行请求存储；缺省进程内实现。
        """
        self._llm = llm
        self._registry = registry
        self._executor = executor
        self._gate = gate
        self._event_stream = event_stream
        self._request_store = request_store or InMemoryRunRequestStore()
        self._graphs: dict[AgentKind, CompiledStateGraph[AgentState]] = {
            AgentKind.REACT: build_react_graph(checkpointer),
            AgentKind.PLANNER_EXECUTOR: build_planner_graph(checkpointer),
        }
        self._cancel_events: dict[UUID, asyncio.Event] = {}

    # ------------------------------------------------------------------ #
    # AgentRuntime 接口
    # ------------------------------------------------------------------ #

    async def run_stream(self, request: AgentRunRequest) -> AsyncIterator[Any]:
        """启动新运行。见 :meth:`src.agents.base.AgentRuntime.run_stream`。"""
        await self._request_store.save(request)
        emitter = EventEmitter(
            request.run_id,
            start_seq=0,
            event_stream=self._event_stream,
            emit_reasoning=request.emit_reasoning,
        )
        initial_state = AgentState(
            run_id=request.run_id,
            conversation_id=request.conversation_id,
            messages=[ChatMessage(role=ChatRole.USER, content=request.input)],
        )
        await emitter.run_started(agent=request.agent.value, model=request.model)
        async for event in self._drive(request, emitter, initial_state):
            yield event

    async def resume_stream(
        self, run_id: UUID, *, approval_id: UUID
    ) -> AsyncIterator[Any]:
        """审批后恢复运行。见 :meth:`src.agents.base.AgentRuntime.resume_stream`。"""
        approval = await self._gate.load(approval_id)
        if approval.run_id != run_id:
            raise NotFoundError(
                "approval does not belong to this run",
                details={"run_id": str(run_id), "approval_id": str(approval_id)},
            )
        if approval.status == "pending":
            raise ConflictError(
                "approval is not resolved yet",
                details={"approval_id": str(approval_id)},
            )
        request = await self._request_store.load(run_id)

        start_seq = 0
        if self._event_stream is not None:
            start_seq = await self._event_stream.last_seq(run_id) + 1
        emitter = EventEmitter(
            run_id,
            start_seq=start_seq,
            event_stream=self._event_stream,
            emit_reasoning=request.emit_reasoning,
        )
        await emitter.approval_resolved(approval)
        resume_command: Command[Any] = Command(
            resume={"approved": approval.status == "approved"}
        )
        async for event in self._drive(request, emitter, resume_command):
            yield event

    async def cancel(self, run_id: UUID, *, principal: Principal) -> None:
        """取消运行。见 :meth:`src.agents.base.AgentRuntime.cancel`。

        阶段 3 为进程内协作取消（图在下一个节点边界终止）；
        跨进程取消标记在阶段 5 接入 Redis。
        """
        request = await self._request_store.load(run_id)
        if request.principal.tenant_id != principal.tenant_id:
            raise NotFoundError("run not found", details={"run_id": str(run_id)})
        cancel_event = self._cancel_events.get(run_id)
        if cancel_event is None:
            raise ConflictError(
                "run is not active", details={"run_id": str(run_id)}
            )
        cancel_event.set()
        _logger.info("run_cancel_requested", run_id=str(run_id), by=principal.id)

    # ------------------------------------------------------------------ #
    # 内部驱动
    # ------------------------------------------------------------------ #

    async def _drive(
        self,
        request: AgentRunRequest,
        emitter: EventEmitter,
        graph_input: AgentState | Command[Any],
    ) -> AsyncIterator[Any]:
        """启动图执行任务并从队列产出事件，直至哨兵。"""
        cancel_event = self._cancel_events.setdefault(request.run_id, asyncio.Event())
        deps = GraphDeps(
            llm=self._llm,
            registry=self._registry,
            executor=self._executor,
            gate=self._gate,
            emitter=emitter,
            request=request,
            cancel_event=cancel_event,
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": str(request.run_id), "deps": deps}
        }
        graph = self._graphs[request.agent]
        task = asyncio.create_task(
            self._execute(graph, graph_input, config, emitter, request)
        )
        try:
            while True:
                event = await emitter.queue.get()
                if event is None:
                    break
                yield event
            await task
        finally:
            if not task.done():
                task.cancel()
            self._cancel_events.pop(request.run_id, None)

    async def _execute(
        self,
        graph: CompiledStateGraph[AgentState],
        graph_input: AgentState | Command[Any],
        config: RunnableConfig,
        emitter: EventEmitter,
        request: AgentRunRequest,
    ) -> None:
        """执行图并把结果翻译为终态事件；任何路径都以哨兵收尾。"""
        started = time.perf_counter()
        try:
            try:
                async with asyncio.timeout(request.guardrails.run_timeout_seconds):
                    result: dict[str, Any] = await graph.ainvoke(graph_input, config)
            except TimeoutError:
                await self._fail(
                    emitter,
                    request,
                    GraphExecutionError(
                        "run timed out",
                        details={
                            "timeout_seconds": request.guardrails.run_timeout_seconds
                        },
                    ),
                )
                return
            except AgentPlatformError as exc:
                _logger.error(
                    "run_failed", run_id=str(request.run_id), code=exc.code
                )
                await self._fail(emitter, request, exc)
                return
            except Exception as exc:
                _logger.exception("run_crashed", run_id=str(request.run_id))
                await self._fail(
                    emitter,
                    request,
                    GraphExecutionError(f"graph execution failed: {exc}", cause=exc),
                )
                return

            if result.get("__interrupt__"):
                # 审批暂停：approval_required 已由 act 节点发出，checkpoint 已落盘
                AGENT_RUNS_TOTAL.labels(
                    agent=request.agent.value, status="waiting_approval"
                ).inc()
                _logger.info("run_waiting_approval", run_id=str(request.run_id))
                return

            termination = result.get("termination_reason")
            if termination == TERMINATION_CANCELLED:
                await self._fail(
                    emitter, request, RunCancelledError("run cancelled by caller")
                )
                return
            if termination:
                await self._fail(
                    emitter,
                    request,
                    GuardrailViolationError(
                        f"run terminated by guardrail: {termination}",
                        details={"termination_reason": termination},
                    ),
                )
                return

            usage = result.get("usage") or TokenUsage()
            await emitter.message_completed(result.get("final_answer") or "")
            await emitter.run_finished(
                usage=usage,
                duration_ms=round((time.perf_counter() - started) * 1000),
            )
            AGENT_RUNS_TOTAL.labels(
                agent=request.agent.value, status="succeeded"
            ).inc()
        finally:
            await emitter.close()

    async def _fail(
        self,
        emitter: EventEmitter,
        request: AgentRunRequest,
        error: AgentPlatformError,
    ) -> None:
        """发出 run_failed 事件并记录指标。"""
        await emitter.run_failed(error=error.to_problem())
        AGENT_RUNS_TOTAL.labels(agent=request.agent.value, status="failed").inc()
