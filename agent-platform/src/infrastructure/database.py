"""PostgreSQL 异步访问：引擎、会话工厂与 UnitOfWork 实现。"""

from __future__ import annotations

from types import TracebackType

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import DatabaseSettings
from src.core.exceptions import DatabaseError


class Database:
    """引擎与会话工厂的生命周期容器（进程级单例，组合根持有）。"""

    def __init__(self, settings: DatabaseSettings) -> None:
        """初始化引擎。

        Args:
            settings: 数据库配置。
        """
        self._engine: AsyncEngine = create_async_engine(
            settings.dsn,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            pool_timeout=settings.pool_timeout_seconds,
            pool_recycle=settings.pool_recycle_seconds,
            pool_pre_ping=True,
            echo=settings.echo,
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False, autoflush=False
        )

    @property
    def engine(self) -> AsyncEngine:
        """底层引擎（迁移与健康检查用）。"""
        return self._engine

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """会话工厂（注入 UnitOfWork 与仓储）。"""
        return self._session_factory

    async def ping(self) -> bool:
        """连通性检查（``/readyz`` 用）。

        Returns:
            True 表示可用；异常时返回 False 而非抛出。
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            return True
        except (SQLAlchemyError, OSError):
            # asyncpg 的连接期错误可能不经 SQLAlchemy 包装直接抛 OSError
            return False

    async def dispose(self) -> None:
        """关闭连接池（进程退出时调用）。"""
        await self._engine.dispose()


class SqlUnitOfWork:
    """:class:`~src.infrastructure.base.UnitOfWork` 的 SQLAlchemy 实现。

    每个实例对应一次事务；``session`` 属性供同一事务内的仓储共享。
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """初始化（不开启会话，进入上下文时才创建）。

        Args:
            session_factory: 会话工厂。
        """
        self._session_factory = session_factory
        self._session: AsyncSession | None = None
        self._committed = False

    @property
    def session(self) -> AsyncSession:
        """当前事务会话。

        Raises:
            DatabaseError: 在上下文之外访问。
        """
        if self._session is None:
            raise DatabaseError("unit of work is not active")
        return self._session

    async def __aenter__(self) -> SqlUnitOfWork:
        """开启会话与事务。"""
        self._session = self._session_factory()
        self._committed = False
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """关闭会话；未提交（或发生异常）则回滚。"""
        assert self._session is not None
        try:
            if exc is not None or not self._committed:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None

    async def commit(self) -> None:
        """提交事务。

        Raises:
            DatabaseError: 提交失败（已回滚）。
        """
        try:
            await self.session.commit()
            self._committed = True
        except SQLAlchemyError as exc:
            await self.session.rollback()
            raise DatabaseError("commit failed", cause=exc) from exc

    async def rollback(self) -> None:
        """显式回滚。"""
        await self.session.rollback()
