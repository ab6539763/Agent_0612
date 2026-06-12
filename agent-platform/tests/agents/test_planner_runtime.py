"""Planner-Executor 图端到端测试。"""

from __future__ import annotations

from typing import Any, cast

from langgraph.checkpoint.memory import MemorySaver
from src.agents.base import AgentKind, ApprovalGate
from src.agents.events import (
    MessageDeltaEvent,
    PlanCreatedEvent,
    PlanUpdatedEvent,
    RunFinishedEvent,
)
from src.agents.graphs.planner import _parse_plan
from src.agents.runtime import LangGraphAgentRuntime
from src.core.types import TokenUsage
from src.llm.base import CompletionChunk, FinishReason, LLMProvider, ToolCallRequest
from src.tools.builtin.calculator import CalculatorTool
from src.tools.executor import DefaultToolExecutor
from src.tools.registry import InMemoryToolRegistry

from tests.agents.fakes import (
    DangerousTool,
    InMemoryApprovalGate,
    ScriptedLLM,
    answer,
    make_principal,
    make_run_request,
    tool_call_response,
)

PLAN_JSON = (
    '[{"description": "compute 6*7", "assignee": "analyst"},'
    ' {"description": "summarize the result", "assignee": "generalist"}]'
)


def respond_stream(*tokens: str) -> list[CompletionChunk]:
    chunks = [CompletionChunk(content_delta=token) for token in tokens]
    chunks.append(
        CompletionChunk(
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=len(tokens)),
        )
    )
    return chunks


def make_runtime(llm: ScriptedLLM) -> LangGraphAgentRuntime:
    registry = InMemoryToolRegistry([CalculatorTool(), DangerousTool()])
    return LangGraphAgentRuntime(
        llm=cast(LLMProvider, llm),
        registry=registry,
        executor=DefaultToolExecutor(registry),
        gate=cast(ApprovalGate, InMemoryApprovalGate()),
        checkpointer=MemorySaver(),
    )


async def collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


class TestPlannerExecutorFlow:
    async def test_full_flow_with_streaming_respond(self) -> None:
        llm = ScriptedLLM(
            completions=[
                answer(PLAN_JSON),  # plan
                tool_call_response(  # executor step 0: 用工具
                    ToolCallRequest(
                        id="c1", name="calculator", arguments={"expression": "6*7"}
                    )
                ),
                answer("step 0 result: 42"),  # executor step 0: 收敛
                answer("step 1 result: result is 42"),  # executor step 1: 直接收敛
                answer("OK"),  # review
            ],
            streams=[respond_stream("The answer ", "is 42.")],
        )
        runtime = make_runtime(llm)
        request = make_run_request(
            make_principal(), agent=AgentKind.PLANNER_EXECUTOR, input_text="what is 6*7?"
        )

        events = await collect(runtime.run_stream(request))

        plan_created = next(e for e in events if isinstance(e, PlanCreatedEvent))
        assert [s.assignee for s in plan_created.steps] == ["analyst", "generalist"]

        updates = [e for e in events if isinstance(e, PlanUpdatedEvent)]
        final_statuses = [s.status for s in updates[-1].steps]
        assert final_statuses == ["done", "done"]

        deltas = [e.delta for e in events if isinstance(e, MessageDeltaEvent)]
        assert deltas == ["The answer ", "is 42."]  # token 级流式

        finished = events[-1]
        assert isinstance(finished, RunFinishedEvent)
        # executor 的角色提示词按 assignee 选择
        executor_request = llm.requests[1]
        assert "data analyst" in executor_request.messages[0].content

    async def test_approval_tools_filtered_out(self) -> None:
        llm = ScriptedLLM(
            completions=[
                answer('[{"description": "do it", "assignee": "generalist"}]'),
                answer("done without dangerous tools"),
                answer("OK"),
            ],
            streams=[respond_stream("done")],
        )
        runtime = make_runtime(llm)
        request = make_run_request(
            make_principal(), agent=AgentKind.PLANNER_EXECUTOR
        )

        events = await collect(runtime.run_stream(request))

        assert isinstance(events[-1], RunFinishedEvent)
        executor_request = llm.requests[1]
        tool_names = {tool.name for tool in executor_request.tools}
        assert "delete_everything" not in tool_names  # 需审批工具被过滤
        assert "calculator" in tool_names

    async def test_replan_once_then_respond(self) -> None:
        llm = ScriptedLLM(
            completions=[
                answer('[{"description": "first try", "assignee": "generalist"}]'),
                answer("insufficient data"),  # executor
                answer("REPLAN: need a calculation step"),  # review → replan
                answer('[{"description": "compute", "assignee": "analyst"}]'),  # plan #2
                answer("computed: 42"),  # executor
                answer("OK"),  # review #2
            ],
            streams=[respond_stream("42")],
        )
        runtime = make_runtime(llm)
        request = make_run_request(
            make_principal(), agent=AgentKind.PLANNER_EXECUTOR
        )

        events = await collect(runtime.run_stream(request))

        assert isinstance(events[-1], RunFinishedEvent)
        # 第二次 plan 收到了评审反馈
        second_plan_request = llm.requests[3]
        assert any(
            "Reviewer feedback" in m.content for m in second_plan_request.messages
        )

    async def test_replan_budget_exhausted_goes_to_respond(self) -> None:
        llm = ScriptedLLM(
            completions=[
                answer('[{"description": "s1", "assignee": "generalist"}]'),
                answer("r1"),
                answer("REPLAN: more"),  # 第 1 次 replan
                answer('[{"description": "s2", "assignee": "generalist"}]'),
                answer("r2"),
                answer("REPLAN: even more"),  # 超出预算 → 仍进入 respond
            ],
            streams=[respond_stream("best effort")],
        )
        runtime = make_runtime(llm)
        events = await collect(
            runtime.run_stream(
                make_run_request(make_principal(), agent=AgentKind.PLANNER_EXECUTOR)
            )
        )
        assert isinstance(events[-1], RunFinishedEvent)


class TestPlanParsing:
    def test_valid_json_array(self) -> None:
        steps = _parse_plan(PLAN_JSON)
        assert len(steps) == 2
        assert steps[0].assignee == "analyst"
        assert steps[1].index == 1

    def test_json_embedded_in_prose(self) -> None:
        content = f"Here is my plan:\n{PLAN_JSON}\nLet me know."
        assert len(_parse_plan(content)) == 2

    def test_unknown_assignee_normalized(self) -> None:
        steps = _parse_plan('[{"description": "x", "assignee": "wizard"}]')
        assert steps[0].assignee == "generalist"

    def test_garbage_falls_back_to_single_step(self) -> None:
        steps = _parse_plan("I will just do the task directly.")
        assert len(steps) == 1
        assert steps[0].assignee == "generalist"

    def test_empty_descriptions_skipped(self) -> None:
        steps = _parse_plan('[{"description": ""}, {"description": "real"}]')
        assert len(steps) == 1
        assert steps[0].description == "real"
