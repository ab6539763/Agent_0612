"""内置工具测试。"""

from __future__ import annotations

from uuid import uuid4

import httpx
import pytest
import respx
from src.core.exceptions import ToolExecutionError
from src.core.types import Principal, PrincipalType, ToolPermission
from src.tools.base import ToolContext
from src.tools.builtin import builtin_tools
from src.tools.builtin.calculator import CalculatorArgs, CalculatorTool
from src.tools.builtin.clock import CurrentTimeArgs, CurrentTimeTool
from src.tools.builtin.http_fetch import HttpFetchArgs, HttpFetchTool


@pytest.fixture()
def context() -> ToolContext:
    return ToolContext(
        principal=Principal(
            id="u",
            type=PrincipalType.USER,
            tenant_id=uuid4(),
            permissions=frozenset({ToolPermission.READ, ToolPermission.NETWORK}),
        ),
        run_id=uuid4(),
    )


class TestCalculator:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("1 + 2 * 3", "7"),
            ("(10 - 4) / 3", "2"),
            ("2 ** 10", "1024"),
            ("-7 % 3", "2"),
            ("17 // 5", "3"),
        ],
    )
    async def test_arithmetic(
        self, context: ToolContext, expression: str, expected: str
    ) -> None:
        out = await CalculatorTool().run(CalculatorArgs(expression=expression), context)
        assert out.endswith(f"= {expected}")

    @pytest.mark.parametrize(
        "expression",
        [
            "__import__('os').system('id')",
            "open('/etc/passwd')",
            "a + 1",
            "'x' * 3",
            "[1,2][0]",
        ],
    )
    async def test_code_injection_rejected(
        self, context: ToolContext, expression: str
    ) -> None:
        with pytest.raises(ToolExecutionError):
            await CalculatorTool().run(CalculatorArgs(expression=expression), context)

    async def test_division_by_zero(self, context: ToolContext) -> None:
        with pytest.raises(ToolExecutionError, match="division by zero"):
            await CalculatorTool().run(CalculatorArgs(expression="1 / 0"), context)

    async def test_huge_exponent_rejected(self, context: ToolContext) -> None:
        with pytest.raises(ToolExecutionError, match="exponent"):
            await CalculatorTool().run(
                CalculatorArgs(expression="9 ** 999999"), context
            )


class TestCurrentTime:
    async def test_utc(self, context: ToolContext) -> None:
        out = await CurrentTimeTool().run(CurrentTimeArgs(), context)
        assert "(UTC)" in out
        assert "+00:00" in out

    async def test_named_zone(self, context: ToolContext) -> None:
        out = await CurrentTimeTool().run(
            CurrentTimeArgs(timezone="Asia/Shanghai"), context
        )
        assert "+08:00" in out

    async def test_bad_zone(self, context: ToolContext) -> None:
        with pytest.raises(ToolExecutionError, match="unknown timezone"):
            await CurrentTimeTool().run(
                CurrentTimeArgs(timezone="Mars/Olympus"), context
            )


class TestHttpFetch:
    @respx.mock
    async def test_fetch_text(self, context: ToolContext) -> None:
        respx.get("https://example.com/page").mock(
            return_value=httpx.Response(200, text="hello world")
        )
        tool = HttpFetchTool(client=httpx.AsyncClient())
        out = await tool.run(HttpFetchArgs(url="https://example.com/page"), context)
        assert "hello world" in out
        assert out.startswith("[200")

    @respx.mock
    async def test_upstream_error(self, context: ToolContext) -> None:
        respx.get("https://example.com/missing").mock(
            return_value=httpx.Response(404)
        )
        tool = HttpFetchTool(client=httpx.AsyncClient())
        with pytest.raises(ToolExecutionError, match="HTTP 404"):
            await tool.run(HttpFetchArgs(url="https://example.com/missing"), context)

    @pytest.mark.parametrize(
        "url",
        [
            "ftp://example.com/file",
            "file:///etc/passwd",
            "http://localhost/admin",
            "http://127.0.0.1:8080/",
            "http://10.0.0.5/internal",
            "http://169.254.169.254/latest/meta-data/",
            "http://user:pass@example.com/",
            "http://metadata.google.internal/computeMetadata/v1/",
        ],
    )
    async def test_ssrf_vectors_rejected(self, context: ToolContext, url: str) -> None:
        tool = HttpFetchTool(client=httpx.AsyncClient())
        with pytest.raises(ToolExecutionError):
            await tool.run(HttpFetchArgs(url=url), context)


class TestBuiltinFactory:
    def test_builds_all(self) -> None:
        tools = builtin_tools()
        assert {tool.name for tool in tools} == {
            "calculator",
            "current_time",
            "http_fetch",
        }
