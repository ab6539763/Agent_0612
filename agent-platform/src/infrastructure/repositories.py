"""核心业务表仓储实现（随阶段 3-5 按需扩展方法集）。"""

from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DatabaseError, NotFoundError
from src.infrastructure.models import AgentRunModel


class SqlAgentRunRepository:
    """:class:`~src.infrastructure.base.AgentRunRepository` 的 SQLAlchemy 实现。

    会话由 UnitOfWork 注入，本类不管理事务边界。
    """

    def __init__(self, session: AsyncSession) -> None:
        """初始化仓储。

        Args:
            session: 当前事务会话（来自 ``SqlUnitOfWork.session``）。
        """
        self._session = session

    async def create(
        self, *, run_id: UUID, tenant_id: UUID, payload: dict[str, Any]
    ) -> None:
        """落库新运行。见 :meth:`src.infrastructure.base.AgentRunRepository.create`。"""
        try:
            self._session.add(
                AgentRunModel(
                    id=run_id,
                    tenant_id=tenant_id,
                    conversation_id=payload.get("conversation_id"),
                    agent=str(payload.get("agent", "react")),
                    model=str(payload.get("model", "")),
                    status="queued",
                    input=str(payload.get("input", "")),
                    trace_id=payload.get("trace_id"),
                )
            )
            await self._session.flush()
        except SQLAlchemyError as exc:
            raise DatabaseError("agent run insert failed", cause=exc) from exc

    async def transition(
        self, run_id: UUID, *, from_status: str, to_status: str
    ) -> bool:
        """CAS 状态迁移。见 :meth:`src.infrastructure.base.AgentRunRepository.transition`。"""
        try:
            result = await self._session.execute(
                update(AgentRunModel)
                .where(AgentRunModel.id == run_id, AgentRunModel.status == from_status)
                .values(status=to_status)
            )
        except SQLAlchemyError as exc:
            raise DatabaseError("agent run transition failed", cause=exc) from exc
        return cast("CursorResult[Any]", result).rowcount == 1

    async def get(self, run_id: UUID, *, tenant_id: UUID) -> dict[str, Any]:
        """读取运行记录。见 :meth:`src.infrastructure.base.AgentRunRepository.get`。"""
        try:
            row = await self._session.scalar(
                select(AgentRunModel).where(
                    AgentRunModel.id == run_id, AgentRunModel.tenant_id == tenant_id
                )
            )
        except SQLAlchemyError as exc:
            raise DatabaseError("agent run query failed", cause=exc) from exc
        if row is None:
            raise NotFoundError("run not found", details={"run_id": str(run_id)})
        return {
            "id": row.id,
            "tenant_id": row.tenant_id,
            "conversation_id": row.conversation_id,
            "agent": row.agent,
            "model": row.model,
            "status": row.status,
            "input": row.input,
            "final_answer": row.final_answer,
            "error": row.error,
            "prompt_tokens": row.prompt_tokens,
            "completion_tokens": row.completion_tokens,
            "trace_id": row.trace_id,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
