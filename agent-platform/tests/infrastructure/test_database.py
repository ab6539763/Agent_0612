"""Database 与 SqlUnitOfWork 测试（sqlite+aiosqlite 内存库）。"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from src.core.config import DatabaseSettings
from src.core.exceptions import DatabaseError
from src.infrastructure.database import Database, SqlUnitOfWork


@pytest.fixture()
async def uow_factory():  # type: ignore[no-untyped-def]
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(text("CREATE TABLE items (id INTEGER PRIMARY KEY, name TEXT)"))
    yield factory
    await engine.dispose()


class TestSqlUnitOfWork:
    async def test_commit_persists(self, uow_factory) -> None:  # type: ignore[no-untyped-def]
        async with SqlUnitOfWork(uow_factory) as uow:
            await uow.session.execute(text("INSERT INTO items (name) VALUES ('a')"))
            await uow.commit()

        async with SqlUnitOfWork(uow_factory) as uow:
            count = await uow.session.scalar(text("SELECT count(*) FROM items"))
            assert count == 1

    async def test_no_commit_rolls_back(self, uow_factory) -> None:  # type: ignore[no-untyped-def]
        async with SqlUnitOfWork(uow_factory) as uow:
            await uow.session.execute(text("INSERT INTO items (name) VALUES ('b')"))
            # 故意不 commit

        async with SqlUnitOfWork(uow_factory) as uow:
            count = await uow.session.scalar(text("SELECT count(*) FROM items"))
            assert count == 0

    async def test_exception_rolls_back(self, uow_factory) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(RuntimeError, match="boom"):
            async with SqlUnitOfWork(uow_factory) as uow:
                await uow.session.execute(text("INSERT INTO items (name) VALUES ('c')"))
                raise RuntimeError("boom")

        async with SqlUnitOfWork(uow_factory) as uow:
            count = await uow.session.scalar(text("SELECT count(*) FROM items"))
            assert count == 0

    async def test_session_outside_context_rejected(self, uow_factory) -> None:  # type: ignore[no-untyped-def]
        uow = SqlUnitOfWork(uow_factory)
        with pytest.raises(DatabaseError, match="not active"):
            _ = uow.session

    async def test_explicit_rollback(self, uow_factory) -> None:  # type: ignore[no-untyped-def]
        async with SqlUnitOfWork(uow_factory) as uow:
            await uow.session.execute(text("INSERT INTO items (name) VALUES ('d')"))
            await uow.rollback()
            count = await uow.session.scalar(text("SELECT count(*) FROM items"))
            assert count == 0


class TestDatabase:
    async def test_ping_unreachable_returns_false(self) -> None:
        db = Database(
            DatabaseSettings(
                dsn="postgresql+asyncpg://nobody:nothing@127.0.0.1:1/void",
                pool_timeout_seconds=0.5,
            )
        )
        assert await db.ping() is False
        await db.dispose()
