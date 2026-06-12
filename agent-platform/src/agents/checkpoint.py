"""LangGraph checkpointer 工厂（ADR-0002）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from src.core.config import DatabaseSettings


def _to_psycopg_dsn(sqlalchemy_dsn: str) -> str:
    """把 SQLAlchemy DSN（``postgresql+asyncpg://``）转为 psycopg 连接串。"""
    return sqlalchemy_dsn.replace("postgresql+asyncpg://", "postgresql://", 1)


@asynccontextmanager
async def postgres_checkpointer(
    settings: DatabaseSettings,
) -> AsyncIterator[AsyncPostgresSaver]:
    """创建 PostgreSQL checkpointer（生命周期与进程一致，组合根持有）。

    首次进入时自动建表（``setup`` 幂等）；checkpoint 表由 LangGraph 官方
    实现自管，不进 Alembic 迁移。

    Args:
        settings: 数据库配置（与业务库同实例）。

    Yields:
        已就绪的 AsyncPostgresSaver。
    """
    async with AsyncPostgresSaver.from_conn_string(
        _to_psycopg_dsn(settings.dsn)
    ) as saver:
        await saver.setup()
        yield saver
