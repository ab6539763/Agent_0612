"""计算器工具：AST 白名单求值，杜绝任意代码执行。"""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import ClassVar

from pydantic import BaseModel, Field

from src.core.exceptions import ToolExecutionError
from src.core.types import ToolPermission
from src.tools.base import BaseTool, ToolContext

_BINARY_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_POW_EXPONENT = 1000
_MAX_EXPRESSION_LENGTH = 500


def _evaluate(node: ast.expr) -> float:
    """递归求值白名单 AST 节点。

    Raises:
        ToolExecutionError: 表达式包含白名单之外的语法。
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        raise ToolExecutionError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(f"operator not allowed: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POW_EXPONENT:
            raise ToolExecutionError("exponent too large")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPS.get(type(node.op))
        if unary is None:
            raise ToolExecutionError(f"operator not allowed: {type(node.op).__name__}")
        return unary(_evaluate(node.operand))
    raise ToolExecutionError(f"syntax not allowed: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    """计算器参数。"""

    expression: str = Field(
        max_length=_MAX_EXPRESSION_LENGTH,
        description="算术表达式，仅支持数字与 + - * / // % ** 及括号。",
    )


class CalculatorTool(BaseTool):
    """安全算术计算器。"""

    name: ClassVar[str] = "calculator"
    description: ClassVar[str] = (
        "计算算术表达式的精确结果。支持加减乘除、整除、取模、幂运算与括号；"
        "不支持变量、函数与字符串。"
    )
    args_schema: ClassVar[type[BaseModel]] = CalculatorArgs
    required_permission: ClassVar[ToolPermission] = ToolPermission.READ

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        """求值表达式。

        Raises:
            ToolExecutionError: 表达式非法或求值错误（除零等）。
        """
        assert isinstance(args, CalculatorArgs)
        try:
            tree = ast.parse(args.expression, mode="eval")
        except SyntaxError as exc:
            raise ToolExecutionError(f"invalid expression: {exc.msg}", cause=exc) from exc
        try:
            result = _evaluate(tree.body)
        except ZeroDivisionError as exc:
            raise ToolExecutionError("division by zero", cause=exc) from exc
        except OverflowError as exc:
            raise ToolExecutionError("result too large", cause=exc) from exc
        return f"{args.expression} = {result:g}"
