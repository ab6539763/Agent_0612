"""ToolRegistry 与 ToolExecutor 测试。"""

from __future__ import annotations

import asyncio
from typing import ClassVar
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field
from src.core.exceptions import ConflictError, ToolExecutionError, ToolNotFoundError
from src.core.types import Principal, PrincipalType, ToolPermission
from src.tools.base import BaseTool, ToolCallInvocation, ToolContext
from src.tools.executor import DefaultToolExecutor
from src.tools.registry import InMemoryToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool):
    name: ClassVar[str] = "echo"
    description: ClassVar[str] = "echo text back"
    args_schema: ClassVar[type[BaseModel]] = EchoArgs
    required_permission: ClassVar[ToolPermission] = ToolPermission.READ

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        assert isinstance(args, EchoArgs)
        return f"echo: {args.text}"


class NetworkTool(EchoTool):
    name: ClassVar[str] = "network_op"
    required_permission: ClassVar[ToolPermission] = ToolPermission.NETWORK


class CrashingTool(EchoTool):
    name: ClassVar[str] = "crashes"

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        raise ToolExecutionError("backend exploded")


class SlowTool(EchoTool):
    name: ClassVar[str] = "slow"

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        await asyncio.sleep(5)
        return "never"


class VerboseTool(EchoTool):
    name: ClassVar[str] = "verbose"

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        return "x" * 10_000


def reader_principal() -> Principal:
    return Principal(
        id="u1",
        type=PrincipalType.USER,
        tenant_id=uuid4(),
        permissions=frozenset({ToolPermission.READ}),
    )


class TestRegistry:
    def test_register_and_get(self) -> None:
        registry = InMemoryToolRegistry([EchoTool()])
        assert registry.get("echo", reader_principal()).name == "echo"

    def test_duplicate_rejected_unless_replace(self) -> None:
        registry = InMemoryToolRegistry([EchoTool()])
        with pytest.raises(ConflictError):
            registry.register(EchoTool())
        registry.register(EchoTool(), replace=True)  # 不抛错

    def test_unregister(self) -> None:
        registry = InMemoryToolRegistry([EchoTool()])
        registry.unregister("echo")
        with pytest.raises(ToolNotFoundError):
            registry.unregister("echo")

    def test_permission_hides_tool_existence(self) -> None:
        registry = InMemoryToolRegistry([NetworkTool()])
        with pytest.raises(ToolNotFoundError):  # 无 network 权限 → 同"不存在"
            registry.get("network_op", reader_principal())

    def test_list_visible_filters_by_permission(self) -> None:
        registry = InMemoryToolRegistry([EchoTool(), NetworkTool()])
        visible = registry.list_visible(reader_principal())
        assert [tool.name for tool in visible] == ["echo"]


class TestExecutor:
    @pytest.fixture()
    def context(self) -> ToolContext:
        return ToolContext(
            principal=reader_principal(), run_id=uuid4(), timeout_seconds=0.2
        )

    def make_executor(self, *tools: BaseTool) -> DefaultToolExecutor:
        return DefaultToolExecutor(
            InMemoryToolRegistry(list(tools)), max_result_chars=200
        )

    async def test_success(self, context: ToolContext) -> None:
        executor = self.make_executor(EchoTool())
        result = await executor.execute(
            ToolCallInvocation(
                tool_call_id="c1", tool_name="echo", raw_arguments={"text": "hi"}
            ),
            context,
        )
        assert result.success
        assert result.content == "echo: hi"

    async def test_invalid_arguments_fed_back(self, context: ToolContext) -> None:
        executor = self.make_executor(EchoTool())
        result = await executor.execute(
            ToolCallInvocation(tool_call_id="c1", tool_name="echo", raw_arguments={}),
            context,
        )
        assert not result.success
        assert "invalid arguments" in result.content
        assert "text" in result.content

    async def test_business_error_fed_back(self, context: ToolContext) -> None:
        executor = self.make_executor(CrashingTool())
        result = await executor.execute(
            ToolCallInvocation(
                tool_call_id="c1", tool_name="crashes", raw_arguments={"text": "x"}
            ),
            context,
        )
        assert not result.success
        assert "backend exploded" in result.content

    async def test_timeout_fed_back(self, context: ToolContext) -> None:
        executor = self.make_executor(SlowTool())
        result = await executor.execute(
            ToolCallInvocation(
                tool_call_id="c1", tool_name="slow", raw_arguments={"text": "x"}
            ),
            context,
        )
        assert not result.success
        assert "timed out" in result.content

    async def test_output_truncated(self, context: ToolContext) -> None:
        executor = self.make_executor(VerboseTool())
        result = await executor.execute(
            ToolCallInvocation(
                tool_call_id="c1", tool_name="verbose", raw_arguments={"text": "x"}
            ),
            context,
        )
        assert result.success
        assert len(result.content) < 300
        assert "truncated" in result.content

    async def test_unknown_tool_raises(self, context: ToolContext) -> None:
        executor = self.make_executor(EchoTool())
        with pytest.raises(ToolNotFoundError):
            await executor.execute(
                ToolCallInvocation(
                    tool_call_id="c1", tool_name="ghost", raw_arguments={}
                ),
                context,
            )
