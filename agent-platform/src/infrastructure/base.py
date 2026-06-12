"""基础设施层接口定义。

domain 层通过这些 Protocol 使用基础设施能力，具体实现（SQLAlchemy 会话、
redis-py 客户端、ARQ 连接）在阶段 2 交付，并在组合根装配。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 事务边界
# ---------------------------------------------------------------------------


@runtime_checkable
class UnitOfWork(Protocol):
    """事务边界抽象。

    用法（service 层）::

        async with uow:
            await runs.update_status(...)
            await approvals.create(...)
            await uow.commit()

    离开上下文未 commit 则自动回滚；嵌套使用 SAVEPOINT。
    """

    async def __aenter__(self) -> UnitOfWork:
        """开启事务。"""
        ...

    async def __aexit__(self, *exc_info: object) -> None:
        """关闭事务（未提交则回滚）。"""
        ...

    async def commit(self) -> None:
        """提交事务。

        Raises:
            DatabaseError: 提交失败（已自动回滚）。
        """
        ...

    async def rollback(self) -> None:
        """显式回滚。"""
        ...


# ---------------------------------------------------------------------------
# 缓存与分布式锁
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheClient(Protocol):
    """Redis 缓存的最小接口（JSON 值语义）。"""

    async def get(self, key: str) -> Any | None:
        """读取键值；不存在返回 None。"""
        ...

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """写入键值，可选 TTL。"""
        ...

    async def delete(self, key: str) -> None:
        """删除键（不存在时为空操作）。"""
        ...


@runtime_checkable
class DistributedLock(Protocol):
    """分布式互斥锁（Redis SET NX + 令牌校验释放实现）。

    用于压缩任务、文档摄取等需要跨进程互斥的操作。
    """

    def acquire(
        self, key: str, *, ttl_seconds: float, timeout_seconds: float
    ) -> AbstractAsyncContextManager[None]:
        """获取锁的异步上下文管理器。

        Args:
            key: 锁键（如 'compress:{conversation_id}'）。
            ttl_seconds: 锁自动过期时间（防持有者崩溃死锁）。
            timeout_seconds: 等待获取的最长时间。

        Raises:
            LockAcquisitionError: 等待超时仍未获取（进入上下文时抛出）。
        """
        ...


# ---------------------------------------------------------------------------
# 事件流（SSE 跨进程下发与断线续传，ADR-0007）
# ---------------------------------------------------------------------------


@runtime_checkable
class EventStream(Protocol):
    """按 run 维度的事件流（Redis Stream 实现）。"""

    async def publish(self, run_id: UUID, *, seq: int, payload: dict[str, Any]) -> None:
        """追加一个事件（worker / runtime 侧调用）。"""
        ...

    def subscribe(
        self, run_id: UUID, *, after_seq: int = -1
    ) -> AsyncIterator[dict[str, Any]]:
        """订阅事件流（API 侧调用）。

        Args:
            run_id: 运行 ID。
            after_seq: 从该序号之后开始读（断线重连传 Last-Event-ID）。

        Yields:
            事件 JSON 载荷；流在终态事件后结束。
        """
        ...


# ---------------------------------------------------------------------------
# 任务队列（ADR-0004）
# ---------------------------------------------------------------------------


class TaskInfo(BaseModel):
    """已入队任务的句柄信息。"""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(description="队列任务 ID（兼作幂等键）。")
    name: str
    enqueued_at: datetime


@runtime_checkable
class TaskQueue(Protocol):
    """任务队列接口（ARQ 实现；业务状态机在 PG，本接口只负责触发执行）。"""

    async def enqueue(
        self,
        name: str,
        *,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        delay_seconds: float = 0.0,
    ) -> TaskInfo:
        """投递任务。

        Args:
            name: 已注册的任务函数名。
            payload: JSON 可序列化的任务参数。
            idempotency_key: 幂等键；相同键的在途任务不会重复入队。
            delay_seconds: 延迟执行秒数。

        Returns:
            任务句柄。

        Raises:
            QueueError: 投递失败。
        """
        ...


# ---------------------------------------------------------------------------
# 仓储（核心业务表的访问接口；完整方法集随阶段 2 模型定义细化）
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentRunRepository(Protocol):
    """``agent_runs`` 表仓储。"""

    async def create(self, *, run_id: UUID, tenant_id: UUID, payload: dict[str, Any]) -> None:
        """落库一次新运行（status=queued）。"""
        ...

    async def transition(
        self, run_id: UUID, *, from_status: str, to_status: str
    ) -> bool:
        """原子状态迁移（CAS）。

        Returns:
            True 表示迁移成功；False 表示当前状态不是 ``from_status``
            （并发竞争，调用方决定重读或放弃）。
        """
        ...

    async def get(self, run_id: UUID, *, tenant_id: UUID) -> dict[str, Any]:
        """读取运行记录。

        Raises:
            NotFoundError: 运行不存在或跨租户。
        """
        ...
