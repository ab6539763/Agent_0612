"""记忆系统接口定义（ADR-0005）。

- :class:`ShortTermMemory`：会话级消息窗口（Redis 实现，阶段 4）。
- :class:`LongTermMemory`：跨会话语义记忆（pgvector 实现，阶段 4）。
- :class:`MemoryCompressor`：渐进式摘要压缩（后台 ARQ 任务调用，阶段 4）。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.llm.base import ChatMessage

# ---------------------------------------------------------------------------
# 短期记忆
# ---------------------------------------------------------------------------


class ConversationWindow(BaseModel):
    """短期记忆读取结果：摘要 + 预算内的消息窗口。"""

    model_config = ConfigDict(frozen=True)

    summary: str | None = Field(
        default=None, description="历史消息的渐进式摘要（无压缩历史时为 None）。"
    )
    messages: tuple[ChatMessage, ...] = Field(
        description="按时间升序、总 token 不超过预算的最近消息。"
    )
    total_message_count: int = Field(
        ge=0, description="会话累计消息数（含已被压缩的）。"
    )
    version: int = Field(
        ge=0,
        description="会话版本号，每次写入递增；压缩任务以此做乐观并发控制。",
    )


@runtime_checkable
class ShortTermMemory(Protocol):
    """会话级短期记忆接口。

    实现要求：append O(1)；read 按 token 预算从最新往回取；
    会话空闲 TTL 自动过期；后端错误翻译为 ``MemoryStoreError``。
    """

    async def append(self, conversation_id: UUID, message: ChatMessage) -> int:
        """追加一条消息。

        Args:
            conversation_id: 会话 ID。
            message: 待追加消息。

        Returns:
            写入后的会话版本号。
        """
        ...

    async def read_window(
        self, conversation_id: UUID, *, token_budget: int
    ) -> ConversationWindow:
        """读取预算内的上下文窗口（摘要 + 最近消息）。

        Args:
            conversation_id: 会话 ID。
            token_budget: 窗口消息的最大 token 数（不含摘要）。

        Returns:
            会话窗口；会话不存在时返回空窗口而非抛错。
        """
        ...

    async def replace_compacted(
        self,
        conversation_id: UUID,
        *,
        summary: str,
        drop_before_index: int,
        expected_version: int,
    ) -> bool:
        """压缩回写：更新摘要并删除已被摘要覆盖的旧消息。

        由 :class:`MemoryCompressor` 后台任务调用。

        Args:
            conversation_id: 会话 ID。
            summary: 新的渐进式摘要全文。
            drop_before_index: 删除该索引之前的所有消息。
            expected_version: 期望的会话版本号（乐观并发控制）。

        Returns:
            True 表示回写成功；False 表示版本冲突（期间有新写入），
            调用方应放弃本次结果并重新调度压缩。
        """
        ...

    async def token_count(self, conversation_id: UUID) -> int:
        """返回当前未压缩消息的累计 token 数（压缩触发判断用）。"""
        ...


# ---------------------------------------------------------------------------
# 长期记忆
# ---------------------------------------------------------------------------


class MemoryKind(StrEnum):
    """长期记忆条目类型。"""

    EPISODIC = "episodic"
    """事件型：发生过的具体交互/事实片段。"""

    SEMANTIC = "semantic"
    """知识型：从交互中蒸馏的稳定事实。"""

    PREFERENCE = "preference"
    """偏好型：用户偏好与约束（回复风格、禁忌项等）。"""


class MemoryRecord(BaseModel):
    """长期记忆条目。"""

    model_config = ConfigDict(frozen=True)

    id: UUID
    tenant_id: UUID
    user_id: str
    kind: MemoryKind
    content: str
    importance: float = Field(
        ge=0.0, le=1.0, description="重要性评分，参与检索排序与淘汰。"
    )
    created_at: datetime
    source_conversation_id: UUID | None = None


class MemoryHit(BaseModel):
    """长期记忆检索命中。"""

    model_config = ConfigDict(frozen=True)

    record: MemoryRecord
    score: float = Field(description="相似度 × 时间衰减 × 重要性的综合分。")


@runtime_checkable
class LongTermMemory(Protocol):
    """跨会话长期记忆接口（pgvector 实现）。

    所有操作按 ``tenant_id + user_id`` 强隔离。
    """

    async def add(
        self,
        *,
        tenant_id: UUID,
        user_id: str,
        kind: MemoryKind,
        content: str,
        importance: float,
        source_conversation_id: UUID | None = None,
    ) -> MemoryRecord:
        """写入一条记忆（实现负责调用 EmbeddingProvider 向量化）。

        Returns:
            已持久化的记忆条目。
        """
        ...

    async def search(
        self,
        *,
        tenant_id: UUID,
        user_id: str,
        query: str,
        top_k: int = 5,
        kinds: tuple[MemoryKind, ...] | None = None,
    ) -> list[MemoryHit]:
        """语义检索记忆。

        Args:
            tenant_id: 租户。
            user_id: 用户。
            query: 查询文本。
            top_k: 返回条数。
            kinds: 限定记忆类型；None 表示全部。

        Returns:
            按综合分降序的命中列表。
        """
        ...

    async def forget(self, *, tenant_id: UUID, memory_id: UUID) -> None:
        """删除一条记忆（用户数据删除合规要求）。

        Raises:
            NotFoundError: 条目不存在或不属于该租户。
        """
        ...


# ---------------------------------------------------------------------------
# 记忆压缩
# ---------------------------------------------------------------------------


class CompressionOutcome(BaseModel):
    """一次压缩任务的产出。"""

    model_config = ConfigDict(frozen=True)

    summary: str = Field(description="更新后的渐进式摘要全文。")
    distilled_memories: tuple[MemoryRecord, ...] = Field(
        default=(),
        description="从被压缩消息中蒸馏出的长期记忆条目（已持久化）。",
    )
    compacted_message_count: int = Field(ge=0)


@runtime_checkable
class MemoryCompressor(Protocol):
    """记忆压缩接口。

    由 ARQ 后台任务在会话 token 超过高水位时调用；实现必须幂等
    （同一版本重复执行产生相同效果），失败可安全重试。
    """

    async def compress(self, conversation_id: UUID) -> CompressionOutcome | None:
        """压缩一个会话的短期记忆。

        流程：读取窗口 → LLM 生成 ``新摘要 = f(旧摘要, 被挤出消息)`` →
        蒸馏长期记忆 → ``replace_compacted`` 乐观回写。

        Returns:
            压缩产出；若无需压缩或版本冲突放弃，返回 None。

        Raises:
            MemoryCompressionError: LLM 调用或持久化失败（可重试）。
        """
        ...
