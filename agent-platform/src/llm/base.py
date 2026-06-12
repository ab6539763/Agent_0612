"""LLM 适配层接口定义（ADR-0003）。

上层代码（agents / memory / rag）只面向本模块的 Protocol 与模型，
不得 import 任何厂商 SDK 类型。实现位于阶段 2：

- ``OpenAIProvider`` / ``AnthropicProvider``：官方 async SDK 适配。
- ``LiteLLMProvider``：覆盖 vLLM 及其他 OpenAI-compatible 端点。
- ``RetryingProvider`` / ``CircuitBreakerProvider``：弹性装饰器（同接口组合）。
- ``ProviderRouter``：按模型名前缀路由（``openai:`` / ``anthropic:`` / ``vllm:``）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from src.core.types import TokenUsage

# ---------------------------------------------------------------------------
# 消息模型（平台自有，与厂商 wire format 解耦）
# ---------------------------------------------------------------------------


class ChatRole(StrEnum):
    """对话消息角色。"""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallRequest(BaseModel):
    """模型发起的一次工具调用请求（Function Calling 输出）。"""

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="调用 ID，工具结果消息须回填以配对。")
    name: str = Field(description="工具名。")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="已解析为字典的调用参数（Provider 负责解析 JSON 字符串）。",
    )


class ChatMessage(BaseModel):
    """平台统一的对话消息。

    约定：
    - ``role=ASSISTANT`` 且发起工具调用时，``tool_calls`` 非空。
    - ``role=TOOL`` 时必须携带 ``tool_call_id`` 与文本化的工具结果 ``content``。
    """

    model_config = ConfigDict(frozen=True)

    role: ChatRole
    content: str = Field(default="")
    tool_calls: tuple[ToolCallRequest, ...] = Field(default=())
    tool_call_id: str | None = Field(
        default=None, description="role=TOOL 时对应的调用 ID。"
    )
    name: str | None = Field(
        default=None, description="role=TOOL 时的工具名，便于审计。"
    )


class ToolSpec(BaseModel):
    """传给模型的工具声明（Function Calling schema）。

    由 tools 模块的 ``BaseTool.to_spec()`` 导出，llm 模块不依赖 tools。
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    parameters: dict[str, Any] = Field(
        description="参数的 JSON Schema（object 类型）。"
    )


# ---------------------------------------------------------------------------
# 请求 / 响应模型
# ---------------------------------------------------------------------------


class CompletionRequest(BaseModel):
    """一次补全请求的全部输入。"""

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        description="带路由前缀的模型名，如 'openai:gpt-4o'、'vllm:qwen2.5-72b'。"
    )
    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...] = Field(default=())
    tool_choice: Literal["auto", "none", "required"] = "auto"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0)
    stop: tuple[str, ...] = Field(default=())
    timeout_seconds: float = Field(
        default=120.0, gt=0, description="单次调用的硬超时（含流式总时长）。"
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="透传的追踪元数据（run_id、tenant_id 等），用于日志与计费归因。",
    )


class FinishReason(StrEnum):
    """补全终止原因（各厂商语义归一化后）。"""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


class CompletionResult(BaseModel):
    """非流式补全的完整结果。"""

    model_config = ConfigDict(frozen=True)

    message: ChatMessage
    finish_reason: FinishReason
    usage: TokenUsage
    model: str = Field(description="实际服务请求的模型（含路由前缀）。")


class CompletionChunk(BaseModel):
    """流式补全的增量片段。

    约定：
    - 文本增量经 ``content_delta`` 下发。
    - 工具调用在 Provider 内部聚合，**完整解析后**作为单个 chunk 的
      ``tool_call`` 下发（上层无需拼接参数 JSON 片段）。
    - 最后一个 chunk 携带 ``finish_reason`` 与 ``usage``。
    """

    model_config = ConfigDict(frozen=True)

    content_delta: str = Field(default="")
    tool_call: ToolCallRequest | None = None
    finish_reason: FinishReason | None = None
    usage: TokenUsage | None = None


# ---------------------------------------------------------------------------
# Provider 接口
# ---------------------------------------------------------------------------


@runtime_checkable
class LLMProvider(Protocol):
    """聊天补全 Provider 的统一接口。

    实现要求：
    - 全异步，禁止任何阻塞调用。
    - 厂商异常必须翻译为 ``src.core.exceptions.LLMError`` 子类。
    - 必须为每次调用创建 OTel span 并记录 token 用量属性。
    """

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """执行一次非流式补全。

        Args:
            request: 补全请求。

        Returns:
            完整补全结果。

        Raises:
            LLMError: 调用失败时抛出对应子类（见 core.exceptions）。
        """
        ...

    def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """执行一次流式补全。

        Args:
            request: 补全请求。

        Yields:
            增量片段，终止 chunk 携带 finish_reason 与 usage。

        Raises:
            LLMError: 连接建立或流中途失败时抛出对应子类。
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """文本向量化接口，供 rag 与 memory 共用。"""

    @property
    def dimension(self) -> int:
        """输出向量维度（建索引与建表时校验一致性）。"""
        ...

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """批量向量化文本。

        Args:
            texts: 待向量化文本，实现负责按上游限制分批。

        Returns:
            与输入等长、顺序一致的向量列表。

        Raises:
            LLMError: 上游调用失败。
        """
        ...
