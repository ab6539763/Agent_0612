"""工具系统接口定义。

- :class:`BaseTool`：所有工具的抽象基类，参数 schema 用 Pydantic 声明，
  自动导出为 Function Calling 的 :class:`~src.llm.base.ToolSpec`。
- :class:`ToolRegistry`：注册中心接口，支持动态注册与按主体权限过滤。
- :class:`ToolExecutor`：统一执行入口（校验、超时、审计、脱敏）。

实现位于阶段 3。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import Principal, ToolPermission


class ToolContext(BaseModel):
    """注入到工具执行的请求上下文（工具不直接接触全局状态）。"""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    principal: Principal
    run_id: UUID = Field(description="当前 Agent 运行 ID，用于审计与追踪关联。")
    conversation_id: UUID | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)


class ToolResult(BaseModel):
    """工具执行结果，作为观察值回喂给模型。"""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    success: bool
    content: str = Field(
        description="文本化结果（成功为输出、失败为面向模型的错误描述）；"
        "超长内容由执行器按预算截断。"
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="结构化附加产物（文件引用、引用来源等），不进模型上下文，"
        "随 tool_result 事件下发给客户端。",
    )


class BaseTool(ABC):
    """工具抽象基类。

    子类必须声明类属性 ``name`` / ``description`` / ``args_schema``，
    并实现 :meth:`run`。声明示例（阶段 3 提供内置工具实现）::

        class WebSearchTool(BaseTool):
            name = "web_search"
            description = "搜索互联网并返回摘要结果"
            args_schema = WebSearchArgs
            required_permission = ToolPermission.NETWORK
    """

    name: ClassVar[str]
    """工具唯一名（snake_case），即 Function Calling 的函数名。"""

    description: ClassVar[str]
    """面向模型的工具描述，决定模型能否正确选用，须清晰说明用途与边界。"""

    args_schema: ClassVar[type[BaseModel]]
    """参数模型；执行前由执行器校验 LLM 产出的参数。"""

    required_permission: ClassVar[ToolPermission] = ToolPermission.READ
    """执行本工具所需的最低权限。"""

    requires_approval: ClassVar[bool] = False
    """是否触发 Human-in-the-loop：执行前暂停运行等待人工审批。"""

    @abstractmethod
    async def run(self, args: BaseModel, context: ToolContext) -> str:
        """执行工具逻辑。

        Args:
            args: 已通过 ``args_schema`` 校验的参数实例。
            context: 请求上下文（主体、run_id、超时预算）。

        Returns:
            文本化执行结果（将回喂给模型）。

        Raises:
            ToolExecutionError: 业务执行失败（执行器会捕获并转为失败 ToolResult）。
        """
        ...

    def to_spec(self) -> ToolSpecExport:
        """导出 Function Calling 声明。

        Returns:
            含 name/description/JSON Schema 的声明对象，由 agents 层
            转换为 ``src.llm.base.ToolSpec`` 传给模型。
        """
        return ToolSpecExport(
            name=self.name,
            description=self.description,
            parameters=self.args_schema.model_json_schema(),
        )


class ToolSpecExport(BaseModel):
    """``BaseTool.to_spec()`` 的输出（与 llm.ToolSpec 字段对齐但模块解耦）。"""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any]


@runtime_checkable
class ToolRegistry(Protocol):
    """工具注册中心接口。

    实现要求：注册/注销线程安全（asyncio 单线程下保证迭代安全）；
    所有读取接口按主体权限过滤——无权限的工具对模型完全不可见，
    而非调用时才拒绝（降低注入风险面）。
    """

    def register(self, tool: BaseTool, *, replace: bool = False) -> None:
        """注册工具。

        Args:
            tool: 工具实例。
            replace: 同名工具已存在时是否替换。

        Raises:
            ConflictError: 同名工具已存在且 ``replace=False``。
        """
        ...

    def unregister(self, name: str) -> None:
        """注销工具。

        Args:
            name: 工具名。

        Raises:
            ToolNotFoundError: 工具未注册。
        """
        ...

    def get(self, name: str, principal: Principal) -> BaseTool:
        """按名称获取主体可见的工具。

        Args:
            name: 工具名。
            principal: 调用主体。

        Returns:
            工具实例。

        Raises:
            ToolNotFoundError: 工具未注册，或主体无权限（不泄露存在性）。
        """
        ...

    def list_visible(self, principal: Principal) -> list[BaseTool]:
        """列举主体可见的全部工具（用于构造模型的 tools 参数）。"""
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """统一工具执行入口。

    职责：参数校验（ToolArgumentError 转为失败结果回喂模型）、权限复核、
    超时控制（``asyncio.timeout``）、结果截断、审计日志与 OTel span、
    输出脱敏。审批检查不在执行器内——``requires_approval`` 由图的审批
    节点在调用前拦截（ADR-0002）。
    """

    async def execute(
        self,
        call: ToolCallInvocation,
        context: ToolContext,
    ) -> ToolResult:
        """执行一次工具调用。

        Args:
            call: 工具调用请求（名称 + 原始参数）。
            context: 请求上下文。

        Returns:
            执行结果；可预期失败（参数错误、业务失败、超时）以
            ``success=False`` 的结果返回而非抛异常，便于模型自我修正。

        Raises:
            ToolNotFoundError: 工具不存在或不可见。
            ToolPermissionDeniedError: 权限复核失败。
        """
        ...


class ToolCallInvocation(BaseModel):
    """一次待执行的工具调用（来自模型的 ToolCallRequest 的领域表示）。"""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    raw_arguments: dict[str, Any] = Field(default_factory=dict)
