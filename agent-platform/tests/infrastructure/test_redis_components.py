"""Redis 缓存、分布式锁与事件流测试（fakeredis）。"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from fakeredis import aioredis as fakeaioredis
from src.core.config import RedisSettings
from src.core.exceptions import LockAcquisitionError
from src.infrastructure.event_stream import RedisEventStream
from src.infrastructure.redis_client import RedisCacheClient, RedisDistributedLock


@pytest.fixture()
async def redis_client() -> fakeaioredis.FakeRedis:
    client = fakeaioredis.FakeRedis(decode_responses=True)
    yield client
    await client.aclose()


class TestRedisCacheClient:
    async def test_roundtrip(self, redis_client: fakeaioredis.FakeRedis) -> None:
        cache = RedisCacheClient(redis_client)
        await cache.set("k", {"a": 1, "b": ["x", "y"]})
        assert await cache.get("k") == {"a": 1, "b": ["x", "y"]}

    async def test_missing_returns_none(
        self, redis_client: fakeaioredis.FakeRedis
    ) -> None:
        cache = RedisCacheClient(redis_client)
        assert await cache.get("absent") is None

    async def test_delete(self, redis_client: fakeaioredis.FakeRedis) -> None:
        cache = RedisCacheClient(redis_client)
        await cache.set("k", 1)
        await cache.delete("k")
        assert await cache.get("k") is None

    async def test_ttl_set(self, redis_client: fakeaioredis.FakeRedis) -> None:
        cache = RedisCacheClient(redis_client)
        await cache.set("k", "v", ttl_seconds=60)
        assert await redis_client.ttl("cache:k") > 0


class TestRedisDistributedLock:
    async def test_mutual_exclusion(self, redis_client: fakeaioredis.FakeRedis) -> None:
        lock = RedisDistributedLock(redis_client)
        async with lock.acquire("job-1", ttl_seconds=5, timeout_seconds=1):
            with pytest.raises(LockAcquisitionError, match="timed out"):
                async with lock.acquire("job-1", ttl_seconds=5, timeout_seconds=0.2):
                    pass

    async def test_released_after_context(
        self, redis_client: fakeaioredis.FakeRedis
    ) -> None:
        lock = RedisDistributedLock(redis_client)
        async with lock.acquire("job-2", ttl_seconds=5, timeout_seconds=1):
            pass
        # 释放后可立即再次获取
        async with lock.acquire("job-2", ttl_seconds=5, timeout_seconds=0.2):
            pass

    async def test_does_not_release_foreign_lock(
        self, redis_client: fakeaioredis.FakeRedis
    ) -> None:
        lock = RedisDistributedLock(redis_client)
        async with lock.acquire("job-3", ttl_seconds=5, timeout_seconds=1):
            # 模拟 TTL 过期后他人持锁：直接覆盖为他人令牌
            await redis_client.set("lock:job-3", "someone-else-token")
        # 退出上下文不应误删他人的锁
        assert await redis_client.get("lock:job-3") == "someone-else-token"


class TestRedisEventStream:
    @pytest.fixture()
    def stream(self, redis_client: fakeaioredis.FakeRedis) -> RedisEventStream:
        return RedisEventStream(redis_client, RedisSettings())

    async def test_publish_subscribe_until_terminal(
        self, stream: RedisEventStream
    ) -> None:
        run_id = uuid4()
        await stream.publish(run_id, seq=0, payload={"type": "run_started"})
        await stream.publish(run_id, seq=1, payload={"type": "message_delta", "delta": "x"})
        await stream.publish(run_id, seq=2, payload={"type": "run_finished"})

        events = [event async for event in stream.subscribe(run_id)]

        assert [e["type"] for e in events] == [
            "run_started",
            "message_delta",
            "run_finished",
        ]

    async def test_resume_after_seq(self, stream: RedisEventStream) -> None:
        run_id = uuid4()
        for seq, kind in enumerate(["run_started", "message_delta", "run_finished"]):
            await stream.publish(run_id, seq=seq, payload={"type": kind, "seq": seq})

        events = [event async for event in stream.subscribe(run_id, after_seq=0)]

        assert [e["seq"] for e in events] == [1, 2]

    async def test_subscriber_receives_live_events(
        self, stream: RedisEventStream
    ) -> None:
        run_id = uuid4()
        await stream.publish(run_id, seq=0, payload={"type": "run_started"})

        async def publish_later() -> None:
            await asyncio.sleep(0.05)
            await stream.publish(run_id, seq=1, payload={"type": "run_finished"})

        task = asyncio.create_task(publish_later())
        events = [event async for event in stream.subscribe(run_id)]
        await task

        assert [e["type"] for e in events] == ["run_started", "run_finished"]

    async def test_stream_has_ttl(
        self, stream: RedisEventStream, redis_client: fakeaioredis.FakeRedis
    ) -> None:
        run_id = uuid4()
        await stream.publish(run_id, seq=0, payload={"type": "run_started"})
        assert await redis_client.ttl(f"events:{run_id}") > 0
