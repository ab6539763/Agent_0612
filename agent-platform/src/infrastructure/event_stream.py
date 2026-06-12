"""Redis Stream 事件流实现（ADR-0007）。

worker / runtime 把 AgentEvent 写入 ``events:{run_id}``，API 层订阅下发 SSE；
事件自带 ``seq``，订阅方按 ``after_seq`` 过滤实现断线续传。
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID

from redis import asyncio as aioredis
from redis.exceptions import RedisError

from src.core.config import RedisSettings
from src.core.exceptions import CacheError

_TERMINAL_EVENT_TYPES = frozenset({"run_finished", "run_failed"})


class RedisEventStream:
    """:class:`~src.infrastructure.base.EventStream` 的 Redis Stream 实现。"""

    _READ_BLOCK_MILLISECONDS = 5_000
    _MAX_IDLE_SECONDS = 900.0
    """订阅端最长空转时间：超过视为生产端异常消失，结束订阅防连接泄漏。"""

    def __init__(self, client: aioredis.Redis, settings: RedisSettings) -> None:
        """初始化事件流。

        Args:
            client: Redis 连接。
            settings: 提供事件流 TTL 配置。
        """
        self._client = client
        self._ttl_seconds = settings.event_stream_ttl_seconds

    def _key(self, run_id: UUID) -> str:
        return f"events:{run_id}"

    async def publish(self, run_id: UUID, *, seq: int, payload: dict[str, Any]) -> None:
        """追加事件。见 :meth:`src.infrastructure.base.EventStream.publish`。"""
        key = self._key(run_id)
        try:
            pipe = self._client.pipeline()
            pipe.xadd(
                key,
                {"seq": str(seq), "payload": json.dumps(payload, ensure_ascii=False)},
            )
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()
        except RedisError as exc:
            raise CacheError("event publish failed", cause=exc) from exc

    async def subscribe(
        self, run_id: UUID, *, after_seq: int = -1
    ) -> AsyncIterator[dict[str, Any]]:
        """订阅事件流。见 :meth:`src.infrastructure.base.EventStream.subscribe`。"""
        key = self._key(run_id)
        last_stream_id = "0-0"
        idle_budget = self._MAX_IDLE_SECONDS

        while True:
            try:
                batches = await self._client.xread(
                    {key: last_stream_id},
                    count=128,
                    block=self._READ_BLOCK_MILLISECONDS,
                )
            except RedisError as exc:
                raise CacheError("event subscribe failed", cause=exc) from exc

            if not batches:
                idle_budget -= self._READ_BLOCK_MILLISECONDS / 1000
                if idle_budget <= 0:
                    return
                continue
            idle_budget = self._MAX_IDLE_SECONDS

            for _key, entries in batches:
                for stream_id, fields in entries:
                    last_stream_id = stream_id
                    seq = int(fields["seq"])
                    if seq <= after_seq:
                        continue
                    payload: dict[str, Any] = json.loads(fields["payload"])
                    yield payload
                    if payload.get("type") in _TERMINAL_EVENT_TYPES:
                        return
