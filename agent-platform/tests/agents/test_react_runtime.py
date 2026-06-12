"""ReAct 图 + LangGraphAgentRuntime 端到端测试（MemorySaver + 脚本化 LLM）。"""

from __future__ import annotations

import asyncio
from typing import Any, cast
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import MemorySaver
from src.agents.base import AgentRunRequest, ApprovalGate, GuardrailConfig
from src.agents.events import (
    ApprovalRequiredEvent,
    ApprovalResolvedEvent,
    MessageCompletedEvent,
    RunFailedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    ToolResultEvent,
)
from src.agents.runtime import LangGraphAgentRuntime
from src.core.exceptions import LLMServerError, NotFoundError
from src.llm.base import LLMProvider, ToolCallRequest
from src.tools.builtin.calculator import CalculatorTool
from src.tools.executor import DefaultToolExecutor
from src.tools.registry import InMemoryToolRegistry

from tests.agents.fakes import (
    DangerousTool,
    InMemoryApprovalGate,
    InMemoryEventStream,
    ScriptedLLM,
    answer,
    make_principal,
    make_run_request,
    tool_call_response,
)


def make_runtime(
    llm: ScriptedLLM,
    *,
    tools: list[Any] | None = None,
    gate: InMemoryApprovalGate | None = None,
    event_stream: InMemoryEventStream | None = None,
) -> LangGraphAgentRuntime:
    registry = InMemoryToolRegistry(tools or [CalculatorTool()])
    return LangGraphAgentRuntime(
        llm=cast(LLMProvider, llm),
        registry=registry,
        executor=DefaultToolExecutor(registry),
        gate=cast(ApprovalGate, gate or InMemoryApprovalGate()),
        checkpointer=MemorySaver(),
        event_stream=event_stream,
    )


async def collect(stream: Any) -> list[Any]:
    return [event async for event in stream]


def event_types(events: list[Any]) -> list[str]:
    return [event.type for event in events]


class TestDirectAnswer:
    async def test_event_sequence(self) -> None:
        llm = ScriptedLLM([answer("Paris is the capital of France.")])
        runtime = make_runtime(llm)
        request = make_run_request(make_principal(), input_text="capital of France?")

        events = await collect(runtime.run_stream(request))

        assert event_types(events) == [
            "run_started",
            "step_started",
            "step_finished",
            "message_delta",
            "message_completed",
            "run_finished",
        ]
        assert isinstance(events[0], RunStartedEvent)
        completed = next(e for e in events if isinstance(e, MessageCompletedEvent))
        assert completed.content == "Paris is the capital of France."
        finished = events[-1]
        assert isinstance(finished, RunFinishedEvent)
        assert finished.usage.total_tokens == 15
        assert [e.seq for e in events] == list(range(len(events)))

    async def test_events_published_to_stream(self) -> None:
        stream = InMemoryEventStream()
        llm = ScriptedLLM([answer("ok")])
        runtime = make_runtime(llm, event_stream=stream)
        request = make_run_request(make_principal())

        await collect(runtime.run_stream(request))

        payloads = stream.published[request.run_id]
        assert payloads[0]["type"] == "run_started"
        assert payloads[-1]["type"] == "run_finished"


class TestToolLoop:
    async def test_tool_call_and_observation(self) -> None:
        llm = ScriptedLLM(
            [
                tool_call_response(
                    ToolCallRequest(
                        id="c1", name="calculator", arguments={"expression": "6*7"}
                    ),
                    reasoning="I need to compute this.",
                ),
                answer("The result is 42."),
            ]
        )
        runtime = make_runtime(llm)
        events = await collect(runtime.run_stream(make_run_request(make_principal())))

        types = event_types(events)
        assert "reasoning_delta" in types
        assert "tool_call" in types
        tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert tool_result.success
        assert "42" in tool_result.content
        assert isinstance(events[-1], RunFinishedEvent)
        # 第二次 LLM 调用携带了工具观察消息
        second_request = llm.requests[1]
        assert any(m.role.value == "tool" for m in second_request.messages)

    async def test_hallucinated_tool_fed_back(self) -> None:
        llm = ScriptedLLM(
            [
                tool_call_response(
                    ToolCallRequest(id="c1", name="ghost_tool", arguments={})
                ),
                answer("I cannot use that tool."),
            ]
        )
        runtime = make_runtime(llm)
        events = await collect(runtime.run_stream(make_run_request(make_principal())))

        tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not tool_result.success
        assert "unavailable" in tool_result.content
        assert isinstance(events[-1], RunFinishedEvent)


class TestGuardrailsAndFailures:
    async def test_max_iterations_guardrail(self) -> None:
        always_tool = [
            tool_call_response(
                ToolCallRequest(
                    id=f"c{i}", name="calculator", arguments={"expression": "1+1"}
                )
            )
            for i in range(3)
        ]
        llm = ScriptedLLM(list(always_tool))
        runtime = make_runtime(llm)
        request = make_run_request(
            make_principal(), guardrails=GuardrailConfig(max_iterations=2)
        )

        events = await collect(runtime.run_stream(request))

        failed = events[-1]
        assert isinstance(failed, RunFailedEvent)
        assert failed.error["title"] == "guardrail_violation"
        assert failed.error["errors"]["termination_reason"] == "max_iterations"

    async def test_llm_failure_becomes_run_failed(self) -> None:
        llm = ScriptedLLM([LLMServerError("upstream down")])
        runtime = make_runtime(llm)
        events = await collect(runtime.run_stream(make_run_request(make_principal())))

        failed = events[-1]
        assert isinstance(failed, RunFailedEvent)
        assert failed.error["title"] == "llm_server_error"

    async def test_run_timeout(self) -> None:
        llm = ScriptedLLM([answer("late")])
        llm.gate_event = asyncio.Event()  # 永不释放 → 超时
        runtime = make_runtime(llm)
        request = make_run_request(
            make_principal(),
            guardrails=GuardrailConfig(run_timeout_seconds=0.2),
        )
        events = await collect(runtime.run_stream(request))
        failed = events[-1]
        assert isinstance(failed, RunFailedEvent)
        assert "timed out" in failed.error["detail"]


class TestCancellation:
    async def test_cancel_terminates_at_next_boundary(self) -> None:
        # 首个回复发起工具调用 → 循环回到第二个 reason 边界时观察到取消
        llm = ScriptedLLM(
            [
                tool_call_response(
                    ToolCallRequest(
                        id="c1", name="calculator", arguments={"expression": "1+1"}
                    )
                ),
                answer("unused"),
            ]
        )
        llm.gate_event = asyncio.Event()
        runtime = make_runtime(llm)
        principal = make_principal()
        request = make_run_request(principal)

        events: list[Any] = []

        async def consume() -> None:
            async for event in runtime.run_stream(request):
                events.append(event)

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.05)  # 等 run_started + reason 进入等待
        await runtime.cancel(request.run_id, principal=principal)
        llm.gate_event.set()  # 释放 LLM；下一个 reason 边界检查到取消
        await asyncio.wait_for(task, timeout=2)

        failed = events[-1]
        assert isinstance(failed, RunFailedEvent)
        assert failed.error["title"] == "run_cancelled"

    async def test_cancel_unknown_run(self) -> None:
        runtime = make_runtime(ScriptedLLM([]))
        with pytest.raises(NotFoundError):
            await runtime.cancel(uuid4(), principal=make_principal())


class TestHumanInTheLoop:
    def _approval_setup(
        self,
    ) -> tuple[
        ScriptedLLM,
        DangerousTool,
        InMemoryApprovalGate,
        LangGraphAgentRuntime,
        AgentRunRequest,
    ]:
        llm = ScriptedLLM(
            [
                tool_call_response(
                    ToolCallRequest(
                        id="c1",
                        name="delete_everything",
                        arguments={"target": "staging-db"},
                    )
                ),
                answer("Done. staging-db removed."),
            ]
        )
        danger = DangerousTool()
        gate = InMemoryApprovalGate()
        runtime = make_runtime(llm, tools=[danger], gate=gate)
        request = make_run_request(make_principal(), input_text="delete staging db")
        return llm, danger, gate, runtime, request

    async def test_run_pauses_with_approval_required(self) -> None:
        _llm, danger, _gate, runtime, request = self._approval_setup()

        events = await collect(runtime.run_stream(request))

        assert isinstance(events[-1], ApprovalRequiredEvent)
        assert events[-1].tool_name == "delete_everything"
        assert danger.executions == []  # interrupt 先于任何执行

    async def test_resume_approved_executes_tool(self) -> None:
        _llm, danger, gate, runtime, request = self._approval_setup()
        run_events = await collect(runtime.run_stream(request))
        approval_event = run_events[-1]
        assert isinstance(approval_event, ApprovalRequiredEvent)

        await gate.resolve(
            approval_event.approval_id, approved=True, resolver=request.principal
        )
        resume_events = await collect(
            runtime.resume_stream(request.run_id, approval_id=approval_event.approval_id)
        )

        assert isinstance(resume_events[0], ApprovalResolvedEvent)
        assert resume_events[0].approved
        tool_result = next(
            e for e in resume_events if isinstance(e, ToolResultEvent)
        )
        assert tool_result.success
        assert danger.executions == ["staging-db"]  # 恰好执行一次（重放安全）
        assert isinstance(resume_events[-1], RunFinishedEvent)
        # seq 续接：恢复段首个 seq 大于运行段最后 seq —— 无 event_stream 时从 0 重新计数，
        # 此处验证带 event_stream 的路径
        assert resume_events[-1].seq > 0

    async def test_resume_seq_continues_with_event_stream(self) -> None:
        llm = ScriptedLLM(
            [
                tool_call_response(
                    ToolCallRequest(
                        id="c1",
                        name="delete_everything",
                        arguments={"target": "x"},
                    )
                ),
                answer("done"),
            ]
        )
        gate = InMemoryApprovalGate()
        stream = InMemoryEventStream()
        runtime = make_runtime(
            llm, tools=[DangerousTool()], gate=gate, event_stream=stream
        )
        request = make_run_request(make_principal())
        run_events = await collect(runtime.run_stream(request))
        approval_event = run_events[-1]
        assert isinstance(approval_event, ApprovalRequiredEvent)
        last_seq_before = await stream.last_seq(request.run_id)

        await gate.resolve(
            approval_event.approval_id, approved=True, resolver=request.principal
        )
        resume_events = await collect(
            runtime.resume_stream(request.run_id, approval_id=approval_event.approval_id)
        )

        assert resume_events[0].seq == last_seq_before + 1
        all_seqs = [p["seq"] for p in stream.published[request.run_id]]
        assert all_seqs == sorted(all_seqs)
        assert len(set(all_seqs)) == len(all_seqs)

    async def test_resume_rejected_feeds_failure_to_model(self) -> None:
        _llm, danger, gate, runtime, request = self._approval_setup()
        run_events = await collect(runtime.run_stream(request))
        approval_event = run_events[-1]
        assert isinstance(approval_event, ApprovalRequiredEvent)

        await gate.resolve(
            approval_event.approval_id, approved=False, resolver=request.principal
        )
        resume_events = await collect(
            runtime.resume_stream(request.run_id, approval_id=approval_event.approval_id)
        )

        tool_result = next(
            e for e in resume_events if isinstance(e, ToolResultEvent)
        )
        assert not tool_result.success
        assert "REJECTED" in tool_result.content
        assert danger.executions == []  # 拒绝后绝不执行
        assert isinstance(resume_events[-1], RunFinishedEvent)  # 运行继续而非失败

    async def test_resume_with_unresolved_approval_conflicts(self) -> None:
        _llm, _danger, _gate, runtime, request = self._approval_setup()
        run_events = await collect(runtime.run_stream(request))
        approval_event = run_events[-1]
        assert isinstance(approval_event, ApprovalRequiredEvent)

        from src.core.exceptions import ConflictError

        with pytest.raises(ConflictError, match="not resolved"):
            await collect(
                runtime.resume_stream(
                    request.run_id, approval_id=approval_event.approval_id
                )
            )
