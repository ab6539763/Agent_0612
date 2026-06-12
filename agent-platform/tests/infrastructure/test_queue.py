"""ArqTaskQueue 测试（连接池 Mock）。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from arq.connections import ArqRedis
from src.core.config import QueueSettings
from src.core.exceptions import QueueError
from src.infrastructure.queue import ArqTaskQueue


class FakePool:
    """最小 ArqRedis 替身。"""

    def __init__(self, *, job_id: str | None = "job-1", error: Exception | None = None) -> None:
        self._job_id = job_id
        self._error = error
        self.calls: list[dict[str, Any]] = []

    async def enqueue_job(self, name: str, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"name": name, "args": args, **kwargs})
        if self._error:
            raise self._error
        if self._job_id is None:
            return None
        return SimpleNamespace(job_id=self._job_id)


def make_queue(pool: FakePool) -> ArqTaskQueue:
    return ArqTaskQueue(cast(ArqRedis, pool), QueueSettings())


class TestArqTaskQueue:
    async def test_enqueue_returns_task_info(self) -> None:
        pool = FakePool(job_id="abc123")
        info = await make_queue(pool).enqueue("ingest_document", payload={"doc": "1"})

        assert info.task_id == "abc123"
        assert info.name == "ingest_document"
        assert pool.calls[0]["name"] == "ingest_document"
        assert pool.calls[0]["args"] == ({"doc": "1"},)

    async def test_idempotency_key_passed_as_job_id(self) -> None:
        pool = FakePool()
        await make_queue(pool).enqueue(
            "compress_memory", payload={}, idempotency_key="conv-42"
        )
        assert pool.calls[0]["_job_id"] == "conv-42"

    async def test_duplicate_treated_as_dedup_hit(self) -> None:
        pool = FakePool(job_id=None)  # arq 对在途 job_id 返回 None
        info = await make_queue(pool).enqueue(
            "compress_memory", payload={}, idempotency_key="conv-42"
        )
        assert info.task_id == "conv-42"

    async def test_delay_forwarded(self) -> None:
        pool = FakePool()
        await make_queue(pool).enqueue("expire_approvals", payload={}, delay_seconds=30)
        assert pool.calls[0]["_defer_by"] == 30

    async def test_backend_error_wrapped(self) -> None:
        pool = FakePool(error=ConnectionError("redis down"))
        with pytest.raises(QueueError, match="enqueue failed"):
            await make_queue(pool).enqueue("ingest_document", payload={})
