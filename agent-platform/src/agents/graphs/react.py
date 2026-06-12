"""ReAct 推理循环图：``reason → act → reason`` 条件环 + ``respond`` 终结。

Human-in-the-loop：``act`` 节点在执行**任何**工具之前检查审批需求并
``interrupt()``。LangGraph 恢复时重放整个节点，因此审批创建是幂等的
（确定性 UUID，见 SqlApprovalGate），且所有执行都发生在 interrupt 之后，
不会产生重复副作用。审批被拒绝时不终止运行——拒绝作为失败的工具观察
结果回喂模型，让其调整策略。

事件语义：推理文本以整段 ``reasoning_delta`` 下发；最终回答在 ``respond``
节点以单个 ``message_delta`` 下发（reason 节点为单次调用经济性使用非流式
补全；token 级流式见 Planner-Executor 的 respond 节点）。
"""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer, interrupt

from src.agents.base import AgentState
from src.agents.graphs.common import (
    GraphDeps,
    build_completion_request,
    check_guardrails,
    get_deps,
    to_llm_tools,
)
from src.agents.prompts import REACT_SYSTEM_PROMPT
from src.core.exceptions import ToolError
from src.core.logging import get_logger
from src.llm.base import ChatMessage, ChatRole, ToolCallRequest
from src.tools.base import ToolCallInvocation, ToolContext, ToolResult

_logger = get_logger(__name__)

_APPROVAL_TTL_SECONDS = 3600
_REJECTION_FEEDBACK = (
    "the human approver REJECTED this action; do not retry it — "
    "choose another approach or explain the limitation to the user"
)


async def _reason(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """推理节点：决定调用工具还是给出最终回答。"""
    deps = get_deps(config)
    termination = check_guardrails(
        deps, iteration=state.iteration, total_tokens=state.usage.total_tokens
    )
    if termination:
        return {"termination_reason": termination}

    await deps.emitter.step_started("reason")
    visible_tools = deps.registry.list_visible(deps.request.principal)
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content=REACT_SYSTEM_PROMPT),
        *state.messages,
    ]
    result = await deps.llm.complete(
        build_completion_request(deps, messages, tools=to_llm_tools(visible_tools))
    )

    updates: dict[str, Any] = {
        "messages": [*state.messages, result.message],
        "usage": state.usage + result.usage,
        "iteration": state.iteration + 1,
    }
    if result.message.tool_calls:
        await deps.emitter.reasoning_delta(result.message.content)
    else:
        updates["final_answer"] = result.message.content
    await deps.emitter.step_finished("reason")
    return updates


def _route_after_reason(state: AgentState) -> str:
    """reason 之后的路由：终止 / 执行工具 / 产出回答。"""
    if state.termination_reason:
        return "end"
    last = state.messages[-1] if state.messages else None
    if last is not None and last.role is ChatRole.ASSISTANT and last.tool_calls:
        return "act"
    return "respond"


def _approval_payload(calls: list[ToolCallRequest]) -> dict[str, Any]:
    """构造审批请求的参数快照（同时是幂等键的输入）。"""
    return {
        "calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in calls
        ]
    }


async def _act(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """执行节点：审批检查（先于一切执行）→ 顺序执行一批工具调用。"""
    deps = get_deps(config)
    await deps.emitter.step_started("act")
    last = state.messages[-1]
    calls = list(last.tool_calls)

    approval_needed = [
        call for call in calls if _requires_approval(deps, call.name)
    ]
    rejected_ids: set[str] = set()
    if approval_needed:
        approval = await deps.gate.create(
            run_id=deps.request.run_id,
            tenant_id=deps.request.principal.tenant_id,
            tool_name=", ".join(sorted({c.name for c in approval_needed})),
            arguments=_approval_payload(approval_needed),
            ttl_seconds=_APPROVAL_TTL_SECONDS,
        )
        if approval.status == "pending":
            await deps.emitter.approval_required(approval)
        # 首次执行在此暂停（checkpoint 已含上方状态）；恢复时重放本节点，
        # interrupt() 直接返回审批决定。
        decision = interrupt({"approval_id": str(approval.id)})
        if not bool(decision.get("approved")):
            rejected_ids = {call.id for call in approval_needed}

    tool_messages: list[ChatMessage] = []
    for call in calls:
        await deps.emitter.tool_call(
            tool_call_id=call.id, tool_name=call.name, arguments=call.arguments
        )
        result = await _execute_call(deps, call, rejected=call.id in rejected_ids)
        await deps.emitter.tool_result(result)
        content = result.content if result.success else f"ERROR: {result.content}"
        tool_messages.append(
            ChatMessage(
                role=ChatRole.TOOL,
                content=content,
                tool_call_id=call.id,
                name=call.name,
            )
        )

    await deps.emitter.step_finished("act")
    return {"messages": [*state.messages, *tool_messages]}


def _requires_approval(deps: GraphDeps, tool_name: str) -> bool:
    """判断工具是否需要审批（不可见/不存在的工具走执行失败路径，不审批）。"""
    try:
        return deps.registry.get(tool_name, deps.request.principal).requires_approval
    except ToolError:
        return False


async def _execute_call(
    deps: GraphDeps, call: ToolCallRequest, *, rejected: bool
) -> ToolResult:
    """执行单个调用；审批拒绝与"工具不存在"都转为失败观察结果回喂模型。"""
    if rejected:
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            success=False,
            content=_REJECTION_FEEDBACK,
        )
    context = ToolContext(
        principal=deps.request.principal,
        run_id=deps.request.run_id,
        conversation_id=deps.request.conversation_id,
    )
    invocation = ToolCallInvocation(
        tool_call_id=call.id, tool_name=call.name, raw_arguments=call.arguments
    )
    try:
        return await deps.executor.execute(invocation, context)
    except ToolError as exc:
        # 模型臆造的工具名等：回喂错误而非终止运行
        _logger.warning("tool_unavailable", tool=call.name, error=exc.message)
        return ToolResult(
            tool_call_id=call.id,
            tool_name=call.name,
            success=False,
            content=f"tool unavailable: {exc.message}",
        )


async def _respond(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """终结节点：下发最终回答（message_completed 由 runtime 统一发出）。"""
    deps = get_deps(config)
    await deps.emitter.message_delta(state.final_answer or "")
    return {}


def build_react_graph(checkpointer: Checkpointer) -> CompiledStateGraph[AgentState]:
    """编译 ReAct 图。

    Args:
        checkpointer: 状态持久化器（生产 AsyncPostgresSaver / 测试 MemorySaver）。

    Returns:
        可通过 ``ainvoke`` 执行的编译图。
    """
    builder: StateGraph[AgentState] = StateGraph(AgentState)
    builder.add_node("reason", _reason)
    builder.add_node("act", _act)
    builder.add_node("respond", _respond)
    builder.set_entry_point("reason")
    builder.add_conditional_edges(
        "reason",
        _route_after_reason,
        {"act": "act", "respond": "respond", "end": END},
    )
    builder.add_edge("act", "reason")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)
