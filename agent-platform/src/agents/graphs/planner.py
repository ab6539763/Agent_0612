"""Planner-Executor 图：``plan → executor(循环) → review → (replan | respond)``。

多 Agent 协作：计划的每个步骤指派给一个执行者角色（researcher / analyst /
generalist），executor 节点按角色加载不同系统提示词执行；步骤结果写回共享
状态供后续步骤与最终回答使用。

已知限制（文档化）：executor 节点内部是有界工具循环，节点重放语义下循环内
``interrupt()`` 会重复执行先前轮次的工具，因此本图**过滤掉需审批的工具**；
需要 Human-in-the-loop 的任务请使用 ReAct 图（act 节点级审批是重放安全的）。

``respond`` 节点以 token 级流式产出最终回答（message_delta）。
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer

from src.agents.base import AgentState
from src.agents.events import PlanStep
from src.agents.graphs.common import (
    build_completion_request,
    check_guardrails,
    get_deps,
    to_llm_tools,
)
from src.agents.prompts import (
    EXECUTOR_ROLE_PROMPTS,
    EXECUTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESPOND_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
)
from src.core.logging import get_logger
from src.llm.base import ChatMessage, ChatRole
from src.tools.base import ToolCallInvocation, ToolContext

_logger = get_logger(__name__)

_MAX_REPLANS = 1
_MAX_TOOL_ROUNDS_PER_STEP = 4
_DEFAULT_ASSIGNEE = "generalist"


def _parse_plan(content: str) -> list[PlanStep]:
    """解析 Planner 输出的 JSON 计划；解析失败时退化为单步计划。"""
    start, end = content.find("["), content.rfind("]")
    if start != -1 and end > start:
        try:
            raw_steps = json.loads(content[start : end + 1])
            steps = [
                PlanStep(
                    index=i,
                    description=str(item.get("description", "")).strip(),
                    assignee=(
                        str(item.get("assignee", _DEFAULT_ASSIGNEE)).strip().lower()
                        if str(item.get("assignee", "")).strip().lower()
                        in EXECUTOR_ROLE_PROMPTS
                        else _DEFAULT_ASSIGNEE
                    ),
                )
                for i, item in enumerate(raw_steps)
                if isinstance(item, dict) and str(item.get("description", "")).strip()
            ]
            if steps:
                return steps
        except (json.JSONDecodeError, TypeError):
            pass
    _logger.warning("plan_parse_fallback", snippet=content[:200])
    return [
        PlanStep(index=0, description=content.strip() or "complete the task",
                 assignee=_DEFAULT_ASSIGNEE)
    ]


async def _plan(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """规划节点：产出（或修订）步骤计划。"""
    deps = get_deps(config)
    termination = check_guardrails(
        deps, iteration=state.iteration, total_tokens=state.usage.total_tokens
    )
    if termination:
        return {"termination_reason": termination}

    await deps.emitter.step_started("plan")
    system = PLANNER_SYSTEM_PROMPT.format(
        assignees=", ".join(EXECUTOR_ROLE_PROMPTS)
    )
    messages = [ChatMessage(role=ChatRole.SYSTEM, content=system), *state.messages]
    result = await deps.llm.complete(build_completion_request(deps, messages))
    plan = _parse_plan(result.message.content)

    if state.replan_count == 0:
        await deps.emitter.plan_created(tuple(plan))
    else:
        await deps.emitter.plan_updated(tuple(plan))
    await deps.emitter.step_finished("plan")
    return {"plan": plan, "usage": state.usage + result.usage}


def _next_pending(plan: list[PlanStep]) -> PlanStep | None:
    """返回第一个待执行步骤。"""
    return next((step for step in plan if step.status == "pending"), None)


def _step_context(state: AgentState, current: PlanStep) -> str:
    """构造执行者的任务上下文（总任务 + 当前步骤 + 已完成步骤结果）。"""
    user_input = next(
        (m.content for m in state.messages if m.role is ChatRole.USER), ""
    )
    done = "\n".join(
        m.content for m in state.messages if m.content.startswith("[step ")
    )
    parts = [f"Overall task: {user_input}", f"Your step: {current.description}"]
    if done:
        parts.append(f"Results from previous steps:\n{done}")
    return "\n\n".join(parts)


async def _execute_step(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """执行节点：以指派角色的提示词执行当前步骤（有界工具循环）。"""
    deps = get_deps(config)
    termination = check_guardrails(
        deps, iteration=state.iteration, total_tokens=state.usage.total_tokens
    )
    if termination:
        return {"termination_reason": termination}

    current = _next_pending(state.plan)
    if current is None:  # 防御：路由保证存在
        return {}

    step_label = f"executor:{current.index}"
    await deps.emitter.step_started(step_label)
    plan = [
        step.model_copy(update={"status": "running"}) if step.index == current.index
        else step
        for step in state.plan
    ]
    await deps.emitter.plan_updated(tuple(plan))

    role_prompt = EXECUTOR_ROLE_PROMPTS.get(
        current.assignee, EXECUTOR_ROLE_PROMPTS[_DEFAULT_ASSIGNEE]
    )
    # 节点重放语义下循环内 interrupt 不安全 → 过滤需审批工具（见模块文档）
    tools = [
        tool
        for tool in deps.registry.list_visible(deps.request.principal)
        if not tool.requires_approval
    ]
    conversation = [
        ChatMessage(
            role=ChatRole.SYSTEM,
            content=EXECUTOR_SYSTEM_PROMPT.format(role_prompt=role_prompt),
        ),
        ChatMessage(role=ChatRole.USER, content=_step_context(state, current)),
    ]

    usage = state.usage
    outcome = "step did not converge within the tool budget"
    succeeded = False
    for _round in range(_MAX_TOOL_ROUNDS_PER_STEP):
        result = await deps.llm.complete(
            build_completion_request(deps, conversation, tools=to_llm_tools(tools))
        )
        usage = usage + result.usage
        conversation.append(result.message)
        if not result.message.tool_calls:
            outcome = result.message.content
            succeeded = True
            break
        await deps.emitter.reasoning_delta(result.message.content)
        for call in result.message.tool_calls:
            await deps.emitter.tool_call(
                tool_call_id=call.id, tool_name=call.name, arguments=call.arguments
            )
            tool_result = await deps.executor.execute(
                ToolCallInvocation(
                    tool_call_id=call.id, tool_name=call.name, raw_arguments=call.arguments
                ),
                ToolContext(
                    principal=deps.request.principal,
                    run_id=deps.request.run_id,
                    conversation_id=deps.request.conversation_id,
                ),
            )
            await deps.emitter.tool_result(tool_result)
            content = (
                tool_result.content
                if tool_result.success
                else f"ERROR: {tool_result.content}"
            )
            conversation.append(
                ChatMessage(
                    role=ChatRole.TOOL,
                    content=content,
                    tool_call_id=call.id,
                    name=call.name,
                )
            )

    plan = [
        step.model_copy(update={"status": "done" if succeeded else "failed"})
        if step.index == current.index
        else step
        for step in plan
    ]
    await deps.emitter.plan_updated(tuple(plan))
    await deps.emitter.step_finished(step_label)
    record = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=f"[step {current.index}: {current.description}]\n{outcome}",
    )
    return {
        "plan": plan,
        "messages": [*state.messages, record],
        "usage": usage,
        "iteration": state.iteration + 1,
    }


def _route_after_executor(state: AgentState) -> str:
    """executor 之后：终止 / 继续下一步骤 / 进入评审。"""
    if state.termination_reason:
        return "end"
    return "executor" if _next_pending(state.plan) is not None else "review"


async def _review(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """评审节点：结果是否足以回答任务；不足且有预算则触发重新规划。"""
    deps = get_deps(config)
    await deps.emitter.step_started("review")
    summary = "\n".join(
        f"- [{step.status}] {step.description}" for step in state.plan
    )
    results = "\n".join(
        m.content for m in state.messages if m.content.startswith("[step ")
    )
    user_input = next(
        (m.content for m in state.messages if m.role is ChatRole.USER), ""
    )
    review_input = ChatMessage(
        role=ChatRole.USER,
        content=f"Task: {user_input}\n\nPlan:\n{summary}\n\nStep results:\n{results}",
    )
    result = await deps.llm.complete(
        build_completion_request(
            deps,
            [ChatMessage(role=ChatRole.SYSTEM, content=REVIEW_SYSTEM_PROMPT), review_input],
        )
    )
    verdict = result.message.content.strip()
    await deps.emitter.step_finished("review")

    updates: dict[str, Any] = {"usage": state.usage + result.usage}
    if verdict.upper().startswith("REPLAN") and state.replan_count < _MAX_REPLANS:
        _logger.info("review_replan", reason=verdict, replan_count=state.replan_count)
        feedback = ChatMessage(
            role=ChatRole.USER, content=f"Reviewer feedback: {verdict}"
        )
        updates.update(
            {
                "plan": [],  # 清空计划 → 路由回 plan 节点
                "replan_count": state.replan_count + 1,
                "messages": [*state.messages, feedback],
            }
        )
    return updates


def _route_after_review(state: AgentState) -> str:
    """review 之后：重新规划或产出最终回答。"""
    if state.termination_reason:
        return "end"
    return "plan" if not state.plan else "respond"


async def _respond(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """终结节点：token 级流式产出最终回答。"""
    deps = get_deps(config)
    await deps.emitter.step_started("respond")
    messages = [
        ChatMessage(role=ChatRole.SYSTEM, content=RESPOND_SYSTEM_PROMPT),
        *state.messages,
    ]
    parts: list[str] = []
    usage = state.usage
    stream = deps.llm.stream(
        build_completion_request(deps, messages, tool_choice="none")
    )
    async for chunk in stream:
        if chunk.content_delta:
            parts.append(chunk.content_delta)
            await deps.emitter.message_delta(chunk.content_delta)
        if chunk.usage is not None:
            usage = usage + chunk.usage
    await deps.emitter.step_finished("respond")
    return {"final_answer": "".join(parts), "usage": usage}


def _route_after_plan(state: AgentState) -> str:
    """plan 之后：终止或开始执行。"""
    return "end" if state.termination_reason else "executor"


def build_planner_graph(checkpointer: Checkpointer) -> CompiledStateGraph[AgentState]:
    """编译 Planner-Executor 图。

    Args:
        checkpointer: 状态持久化器。

    Returns:
        可通过 ``ainvoke`` 执行的编译图。
    """
    builder: StateGraph[AgentState] = StateGraph(AgentState)
    builder.add_node("plan", _plan)
    builder.add_node("executor", _execute_step)
    builder.add_node("review", _review)
    builder.add_node("respond", _respond)
    builder.set_entry_point("plan")
    builder.add_conditional_edges(
        "plan", _route_after_plan, {"executor": "executor", "end": END}
    )
    builder.add_conditional_edges(
        "executor",
        _route_after_executor,
        {"executor": "executor", "review": "review", "end": END},
    )
    builder.add_conditional_edges(
        "review", _route_after_review, {"plan": "plan", "respond": "respond", "end": END}
    )
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)
