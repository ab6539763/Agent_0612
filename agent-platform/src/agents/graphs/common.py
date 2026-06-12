"""图节点共享的依赖容器与工具函数。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.runnables import RunnableConfig

from src.agents.base import AgentRunRequest, ApprovalGate
from src.agents.emitter import EventEmitter
from src.core.exceptions import GraphExecutionError
from src.llm.base import (
    ChatMessage,
    CompletionRequest,
    LLMProvider,
    ToolSpec,
)
from src.tools.base import BaseTool, ToolExecutor, ToolRegistry

TERMINATION_MAX_ITERATIONS = "max_iterations"
TERMINATION_TOKEN_BUDGET = "token_budget"  # noqa: S105 - 终止原因常量，非凭证
TERMINATION_CANCELLED = "cancelled"


@dataclass(frozen=True)
class GraphDeps:
    """节点运行期依赖（经 ``config["configurable"]["deps"]`` 注入，不进 checkpoint）。"""

    llm: LLMProvider
    registry: ToolRegistry
    executor: ToolExecutor
    gate: ApprovalGate
    emitter: EventEmitter
    request: AgentRunRequest
    cancel_event: asyncio.Event


def get_deps(config: RunnableConfig) -> GraphDeps:
    """从节点 config 提取依赖容器。

    Raises:
        GraphExecutionError: 图未经 runtime 启动（缺少依赖注入）。
    """
    deps = (config.get("configurable") or {}).get("deps")
    if not isinstance(deps, GraphDeps):
        raise GraphExecutionError("graph started without injected GraphDeps")
    return deps


def to_llm_tools(tools: list[BaseTool]) -> tuple[ToolSpec, ...]:
    """把工具声明转换为 LLM 层的 Function Calling spec。"""
    return tuple(
        ToolSpec(
            name=spec.name, description=spec.description, parameters=spec.parameters
        )
        for spec in (tool.to_spec() for tool in tools)
    )


def build_completion_request(
    deps: GraphDeps,
    messages: list[ChatMessage],
    *,
    tools: tuple[ToolSpec, ...] = (),
    tool_choice: Literal["auto", "none", "required"] = "auto",
) -> CompletionRequest:
    """构造带追踪元数据的补全请求。"""
    request = deps.request
    return CompletionRequest(
        model=request.model,
        messages=tuple(messages),
        tools=tools,
        tool_choice=tool_choice,
        metadata={
            "run_id": str(request.run_id),
            "tenant_id": str(request.principal.tenant_id),
        },
    )


def check_guardrails(
    deps: GraphDeps, *, iteration: int, total_tokens: int
) -> str | None:
    """检查护栏，返回终止原因或 None。

    Args:
        deps: 节点依赖。
        iteration: 已完成的循环次数。
        total_tokens: 累计 token 消耗。

    Returns:
        ``max_iterations`` / ``token_budget`` / ``cancelled`` / None。
    """
    guardrails = deps.request.guardrails
    if deps.cancel_event.is_set():
        return TERMINATION_CANCELLED
    if iteration >= guardrails.max_iterations:
        return TERMINATION_MAX_ITERATIONS
    if total_tokens >= guardrails.token_budget:
        return TERMINATION_TOKEN_BUDGET
    return None


def configurable(config: RunnableConfig) -> dict[str, Any]:
    """返回 config 的 configurable 字典（缺省为空）。"""
    return config.get("configurable") or {}
