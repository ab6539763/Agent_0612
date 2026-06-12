"""Redis 客户端、缓存与分布式锁实现。"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import time
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, cast

from redis import asyncio as aioredis
from redis.exceptions import RedisError

from src.core.config import RedisSettings
from src.core.exceptions import CacheError, LockAcquisitionError

_RELEASE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""
"""令牌比对后才删除——防止 TTL 过期后误删他人持有的锁。"""


def create_redis(settings: RedisSettings) -> aioredis.Redis:
    """按配置创建 Redis 客户端（进程级单例，组合根持有）。

    Args:
        settings: Redis 配置。

    Returns:
        decode_responses 模式的异步客户端。
    """
    client: aioredis.Redis = aioredis.Redis.from_url(
        settings.url,
        max_connections=settings.max_connections,
        socket_timeout=settings.socket_timeout_seconds,
        socket_connect_timeout=settings.socket_timeout_seconds,
        decode_responses=True,
    )
    return client


async def ping_redis(client: aioredis.Redis) -> bool:
    """连通性检查（``/readyz`` 用），异常时返回 False。"""
    try:
        return bool(await client.ping())
    except RedisError:
        return False


class RedisCacheClient:
    """:class:`~src.infrastructure.base.CacheClient` 的 Redis 实现（JSON 值）。"""

    def __init__(self, client: aioredis.Redis, *, prefix: str = "cache") -> None:
        """初始化缓存客户端。

        Args:
            client: Redis 连接。
            prefix: 键前缀（与锁、事件流隔离命名空间）。
        """
        self._client = client
        self._prefix = prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    async def get(self, key: str) -> Any | None:
        """读取键值。见 :meth:`src.infrastructure.base.CacheClient.get`。"""
        try:
            raw = await self._client.get(self._key(key))
        except RedisError as exc:
            raise CacheError("redis get failed", cause=exc) from exc
        return None if raw is None else json.loads(raw)

    async def set(self, key: str, value: Any, *, ttl_seconds: int | None = None) -> None:
        """写入键值。见 :meth:`src.infrastructure.base.CacheClient.set`。"""
        try:
            await self._client.set(
                self._key(key), json.dumps(value, ensure_ascii=False), ex=ttl_seconds
            )
        except RedisError as exc:
            raise CacheError("redis set failed", cause=exc) from exc

    async def delete(self, key: str) -> None:
        """删除键。见 :meth:`src.infrastructure.base.CacheClient.delete`。"""
        try:
            await self._client.delete(self._key(key))
        except RedisError as exc:
            raise CacheError("redis delete failed", cause=exc) from exc


class RedisDistributedLock:
    """:class:`~src.infrastructure.base.DistributedLock` 的 Redis 实现。

    ``SET NX PX`` 抢锁 + 随机令牌 + Lua 比对释放；等待期间指数退避轮询。
    """

    _POLL_INITIAL_SECONDS = 0.05
    _POLL_MAX_SECONDS = 0.5

    def __init__(self, client: aioredis.Redis, *, prefix: str = "lock") -> None:
        """初始化锁工厂。

        Args:
            client: Redis 连接。
            prefix: 锁键前缀。
        """
        self._client = client
        self._prefix = prefix

    @asynccontextmanager
    async def acquire(
        self, key: str, *, ttl_seconds: float, timeout_seconds: float
    ) -> AsyncIterator[None]:
        """获取锁。见 :meth:`src.infrastructure.base.DistributedLock.acquire`。"""
        lock_key = f"{self._prefix}:{key}"
        token = secrets.token_hex(16)
        deadline = time.monotonic() + timeout_seconds
        poll = self._POLL_INITIAL_SECONDS

        while True:
            try:
                acquired = await self._client.set(
                    lock_key, token, nx=True, px=int(ttl_seconds * 1000)
                )
            except RedisError as exc:
                raise LockAcquisitionError("redis lock backend failed", cause=exc) from exc
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise LockAcquisitionError(
                    "lock acquisition timed out", details={"key": key}
                )
            await asyncio.sleep(poll)
            poll = min(poll * 2, self._POLL_MAX_SECONDS)

        try:
            yield
        finally:
            # 释放失败由 TTL 兜底过期，不向上扰动业务结果
            with contextlib.suppress(RedisError):
                release = cast(
                    "Awaitable[Any]",
                    self._client.eval(_RELEASE_LOCK_SCRIPT, 1, lock_key, token),
                )
                await release
