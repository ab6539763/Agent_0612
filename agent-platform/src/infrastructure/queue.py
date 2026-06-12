"""ARQ 任务队列实现（ADR-0004）。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from arq import create_pool
from arq.connections import ArqRedis
from arq.connections import RedisSettings as ArqRedisSettings

from src.core.config import QueueSettings
from src.core.exceptions import QueueError
from src.core.logging import get_logger
from src.core.observability import QUEUE_TASKS_TOTAL
from src.infrastructure.base import TaskInfo

_logger = get_logger(__name__)


async def create_task_pool(settings: QueueSettings) -> ArqRedis:
    """创建 ARQ 连接池（组合根与 worker 入口调用）。

    Raises:
        QueueError: Redis 不可达。
    """
    try:
        return await create_pool(
            ArqRedisSettings.from_dsn(settings.redis_url),
            default_queue_name=settings.queue_name,
        )
    except OSError as exc:
        raise QueueError("queue redis unreachable", cause=exc) from exc


class ArqTaskQueue:
    """:class:`~src.infrastructure.base.TaskQueue` 的 ARQ 实现。

    幂等：``idempotency_key`` 直接作为 ARQ ``job_id``——同 ID 的在途任务
    重复投递会被 ARQ 拒绝，本实现将其视为去重命中而非错误。
    """

    def __init__(self, pool: ArqRedis, settings: QueueSettings) -> None:
        """初始化队列适配器。

        Args:
            pool: ARQ 连接池。
            settings: 队列配置。
        """
        self._pool = pool
        self._settings = settings

    async def enqueue(
        self,
        name: str,
        *,
        payload: dict[str, Any],
        idempotency_key: str | None = None,
        delay_seconds: float = 0.0,
    ) -> TaskInfo:
        """投递任务。见 :meth:`src.infrastructure.base.TaskQueue.enqueue`。"""
        try:
            job = await self._pool.enqueue_job(
                name,
                payload,
                _job_id=idempotency_key,
                _defer_by=delay_seconds if delay_seconds > 0 else None,
                _queue_name=self._settings.queue_name,
            )
        except Exception as exc:
            QUEUE_TASKS_TOTAL.labels(task=name, outcome="error").inc()
            raise QueueError("enqueue failed", details={"task": name}, cause=exc) from exc

        if job is None:
            # 同 job_id 在途：幂等去重命中
            assert idempotency_key is not None
            QUEUE_TASKS_TOTAL.labels(task=name, outcome="deduplicated").inc()
            _logger.info("task_deduplicated", task=name, job_id=idempotency_key)
            return TaskInfo(
                task_id=idempotency_key, name=name, enqueued_at=datetime.now(tz=UTC)
            )

        QUEUE_TASKS_TOTAL.labels(task=name, outcome="enqueued").inc()
        return TaskInfo(task_id=job.job_id, name=name, enqueued_at=datetime.now(tz=UTC))
