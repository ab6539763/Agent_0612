"""domain 接口契约测试：模型可序列化、判别联合正确、异常映射完整。"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field, TypeAdapter
from src.agents.base import AgentRunRequest, AgentState, RunStatus
from src.agents.events import (
    AgentEvent,
    ApprovalRequiredEvent,
    RunFailedEvent,
    ToolCallEvent,
)
from src.core.exceptions import (
    AgentPlatformError,
    LLMRateLimitError,
    NotFoundError,
    ToolPermissionDeniedError,
)
from src.core.types import Principal, PrincipalType, TokenUsage, ToolPermission
from src.llm.base import ChatMessage, ChatRole
from src.memory.base import ConversationWindow, MemoryKind
from src.rag.base import DocumentFormat, RetrievalQuery
from src.tools.base import BaseTool, ToolContext


class TestExceptions:
    def test_problem_details_shape(self) -> None:
        exc = LLMRateLimitError("throttled", details={"provider": "openai"})
        problem = exc.to_problem()
        assert problem["status"] == 429
        assert problem["title"] == "llm_rate_limited"
        assert problem["retryable"] is True
        assert problem["errors"] == {"provider": "openai"}

    def test_problem_without_details(self) -> None:
        problem = NotFoundError("run not found").to_problem()
        assert "errors" not in problem
        assert problem["status"] == 404

    def test_cause_chained(self) -> None:
        original = ValueError("root")
        exc = AgentPlatformError("wrapped", cause=original)
        assert exc.__cause__ is original

    def test_subclasses_inherit_machinery(self) -> None:
        exc = ToolPermissionDeniedError("denied")
        assert exc.http_status == 403
        assert not exc.retryable


class TestTokenUsage:
    def test_addition(self) -> None:
        total = TokenUsage(prompt_tokens=10, completion_tokens=5) + TokenUsage(
            prompt_tokens=3, completion_tokens=2
        )
        assert total.prompt_tokens == 13
        assert total.total_tokens == 20


class TestAgentEvents:
    def test_discriminated_union_roundtrip(self) -> None:
        adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)
        event = ToolCallEvent(
            run_id=uuid4(),
            seq=3,
            timestamp=datetime.now(tz=UTC),
            tool_call_id="c1",
            tool_name="search",
            arguments={"q": "redis"},
        )
        restored = adapter.validate_json(event.model_dump_json())
        assert isinstance(restored, ToolCallEvent)
        assert restored.arguments == {"q": "redis"}

    def test_run_failed_carries_problem(self) -> None:
        event = RunFailedEvent(
            run_id=uuid4(),
            seq=9,
            timestamp=datetime.now(tz=UTC),
            error=LLMRateLimitError("x").to_problem(),
        )
        assert event.error["status"] == 429

    def test_approval_event_fields(self) -> None:
        event = ApprovalRequiredEvent(
            run_id=uuid4(),
            seq=1,
            timestamp=datetime.now(tz=UTC),
            approval_id=uuid4(),
            tool_name="delete_records",
            arguments={"table": "users"},
            expires_at=datetime.now(tz=UTC),
        )
        assert event.type == "approval_required"


class TestAgentState:
    def test_state_json_serializable(self) -> None:
        state = AgentState(
            run_id=uuid4(),
            conversation_id=uuid4(),
            messages=[ChatMessage(role=ChatRole.USER, content="hi")],
        )
        restored = AgentState.model_validate_json(state.model_dump_json())
        assert restored.messages[0].content == "hi"
        assert restored.iteration == 0

    def test_run_request_validation(self, principal: Principal) -> None:
        request = AgentRunRequest(
            run_id=uuid4(),
            conversation_id=uuid4(),
            principal=principal,
            input="hello",
            model="openai:gpt-4o",
        )
        assert request.guardrails.max_iterations == 15

    def test_run_status_values(self) -> None:
        assert RunStatus.WAITING_APPROVAL.value == "waiting_approval"


class TestToolContract:
    def test_tool_spec_export(self, principal: Principal) -> None:
        class EchoArgs(BaseModel):
            text: str = Field(description="text to echo")

        class EchoTool(BaseTool):
            name = "echo"
            description = "echo input back"
            args_schema = EchoArgs
            required_permission = ToolPermission.READ

            async def run(self, args: BaseModel, context: ToolContext) -> str:
                assert isinstance(args, EchoArgs)
                return args.text

        spec = EchoTool().to_spec()
        assert spec.name == "echo"
        assert spec.parameters["properties"]["text"]["description"] == "text to echo"

        context = ToolContext(principal=principal, run_id=uuid4())
        assert context.timeout_seconds == 30.0


class TestMemoryAndRagModels:
    def test_conversation_window_defaults(self) -> None:
        window = ConversationWindow(messages=(), total_message_count=0, version=0)
        assert window.summary is None

    def test_memory_kind_values(self) -> None:
        assert {k.value for k in MemoryKind} == {"episodic", "semantic", "preference"}

    def test_retrieval_query_bounds(self) -> None:
        with pytest.raises(ValueError, match="less_than_equal"):
            RetrievalQuery(tenant_id=uuid4(), query="q", top_k=999)

    def test_document_format(self) -> None:
        assert DocumentFormat("pdf") is DocumentFormat.PDF


class TestPrincipal:
    def test_frozen(self, principal: Principal) -> None:
        with pytest.raises(Exception, match="frozen"):
            principal.id = "tampered"  # type: ignore[misc]

    def test_principal_types(self) -> None:
        assert PrincipalType.API_KEY.value == "api_key"
