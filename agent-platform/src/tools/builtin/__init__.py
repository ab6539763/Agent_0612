"""内置工具集。

组合根通过 :func:`builtin_tools` 批量注册；RAG 的 ``knowledge_search``
工具在阶段 4 随检索管道一起加入。
"""

from __future__ import annotations

import httpx

from src.tools.base import BaseTool
from src.tools.builtin.calculator import CalculatorTool
from src.tools.builtin.clock import CurrentTimeTool
from src.tools.builtin.http_fetch import HttpFetchTool

__all__ = ["CalculatorTool", "CurrentTimeTool", "HttpFetchTool", "builtin_tools"]


def builtin_tools(http_client: httpx.AsyncClient | None = None) -> list[BaseTool]:
    """构造全部内置工具实例。

    Args:
        http_client: 注入的 HTTP 客户端（连接池复用；缺省由工具自建）。

    Returns:
        可直接批量注册进 ToolRegistry 的工具列表。
    """
    return [
        CalculatorTool(),
        CurrentTimeTool(),
        HttpFetchTool(client=http_client),
    ]
