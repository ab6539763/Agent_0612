"""EventEmitter 测试。"""

from __future__ import annotations

from uuid import uuid4

from src.agents.emitter import EventEmitter
from src.agents.events import ReasoningDeltaEvent, RunStartedEvent, ToolCallEvent
from src.core.redaction import REDACTED

from tests.agents.fakes import InMemoryEventStream


async def drain(emitter: EventEmitter) -> list[object]:
    await emitter.close()
    events: list[object] = []
    while True:
        event = await emitter.queue.get()
        if event is None:
            break
        events.append(event)
    return events


class TestEventEmitter:
    async def test_seq_monotonic_from_start_seq(self) -> None:
        emitter = EventEmitter(uuid4(), start_seq=7)
        await emitter.run_started(agent="react", model="openai:gpt-4o")
        await emitter.step_started("reason")
        await emitter.message_delta("hi")

        events = await drain(emitter)

        assert [e.seq for e in events] == [7, 8, 9]  # type: ignore[attr-defined]
        assert isinstance(events[0], RunStartedEvent)

    async def test_tool_arguments_redacted(self) -> None:
        emitter = EventEmitter(uuid4())
        await emitter.tool_call(
            tool_call_id="c1",
            tool_name="http_fetch",
            arguments={"url": "https://x.dev", "api_key": "sk-secret-12345678"},
        )
        events = await drain(emitter)
        event = events[0]
        assert isinstance(event, ToolCallEvent)
        assert event.arguments["api_key"] == REDACTED
        assert event.arguments["url"] == "https://x.dev"

    async def test_reasoning_suppressed_when_disabled(self) -> None:
        emitter = EventEmitter(uuid4(), emit_reasoning=False)
        await emitter.reasoning_delta("secret chain of thought")
        await emitter.message_delta("answer")
        events = await drain(emitter)
        assert len(events) == 1
        assert not isinstance(events[0], ReasoningDeltaEvent)

    async def test_publishes_to_event_stream(self) -> None:
        stream = InMemoryEventStream()
        run_id = uuid4()
        emitter = EventEmitter(run_id, event_stream=stream)
        await emitter.run_started(agent="react", model="m")
        await emitter.message_completed("done")

        payloads = stream.published[run_id]
        assert [p["seq"] for p in payloads] == [0, 1]
        assert payloads[0]["type"] == "run_started"
        assert await stream.last_seq(run_id) == 1

    async def test_empty_deltas_skipped(self) -> None:
        emitter = EventEmitter(uuid4())
        await emitter.message_delta("")
        await emitter.reasoning_delta("")
        assert await drain(emitter) == []
