"""进程内工具注册中心。"""

from __future__ import annotations

from src.core.exceptions import ConflictError, ToolNotFoundError
from src.core.logging import get_logger
from src.core.types import Principal
from src.tools.base import BaseTool

_logger = get_logger(__name__)


class InMemoryToolRegistry:
    """:class:`~src.tools.base.ToolRegistry` 的进程内实现。

    权限语义：主体缺少工具所需权限时，该工具对其**完全不可见**——
    ``get`` 统一抛 ``ToolNotFoundError``（不泄露存在性），``list_visible``
    直接过滤，模型的 tools 参数里不会出现无权限工具。
    """

    def __init__(self, tools: list[BaseTool] | None = None) -> None:
        """初始化注册中心。

        Args:
            tools: 启动时批量注册的工具。
        """
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        """注册工具。见 :meth:`src.tools.base.ToolRegistry.register`。"""
        if tool.name in self._tools and not replace:
            raise ConflictError(
                f"tool '{tool.name}' already registered", details={"tool": tool.name}
            )
        self._tools[tool.name] = tool
        _logger.info(
            "tool_registered",
            tool=tool.name,
            permission=tool.required_permission.value,
            requires_approval=tool.requires_approval,
        )

    def unregister(self, name: str) -> None:
        """注销工具。见 :meth:`src.tools.base.ToolRegistry.unregister`。"""
        if name not in self._tools:
            raise ToolNotFoundError(f"tool '{name}' is not registered")
        del self._tools[name]
        _logger.info("tool_unregistered", tool=name)

    def get(self, name: str, principal: Principal) -> BaseTool:
        """获取主体可见的工具。见 :meth:`src.tools.base.ToolRegistry.get`。"""
        tool = self._tools.get(name)
        if tool is None or tool.required_permission not in principal.permissions:
            raise ToolNotFoundError(
                f"tool '{name}' not found", details={"tool": name}
            )
        return tool

    def list_visible(self, principal: Principal) -> list[BaseTool]:
        """列举可见工具。见 :meth:`src.tools.base.ToolRegistry.list_visible`。"""
        return [
            tool
            for tool in self._tools.values()
            if tool.required_permission in principal.permissions
        ]
