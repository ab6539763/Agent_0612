"""Human-in-the-loop 审批网关的 PostgreSQL 实现。

幂等性设计：审批 ID 由 ``uuid5(run_id + 参数快照)`` 确定性生成——LangGraph
恢复时重放 act 节点会再次调用 :meth:`create`，命中已有行直接返回，
不产生重复审批。
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.agents.base import ApprovalRequest
from src.core.exceptions import ConflictError, DatabaseError, NotFoundError
from src.core.logging import get_logger
from src.core.types import Principal
from src.infrastructure.models import ApprovalModel

_logger = get_logger(__name__)

_APPROVAL_NAMESPACE = uuid.UUID("7e6b1cb1-94a8-4d5e-9f70-9a4f6d8a2b11")


def _deterministic_id(run_id: UUID, arguments: dict[str, Any]) -> UUID:
    """从运行 ID 与参数快照派生确定性审批 ID（重放幂等的关键）。"""
    digest = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    return uuid.uuid5(_APPROVAL_NAMESPACE, f"{run_id}:{digest}")


def _now_like(reference: datetime) -> datetime:
    """返回与参考值时区形态一致的当前 UTC 时间（SQLite 存储为 naive）。"""
    now = datetime.now(tz=UTC)
    return now.replace(tzinfo=None) if reference.tzinfo is None else now


def _to_request(row: ApprovalModel) -> ApprovalRequest:
    """ORM 行 → 领域模型。"""
    return ApprovalRequest(
        id=row.id,
        run_id=row.run_id,
        tenant_id=row.tenant_id,
        tool_name=row.tool_name,
        arguments=row.arguments,
        requested_at=row.created_at,
        expires_at=row.expires_at,
        status=row.status,
        resolver=row.resolver,
        resolved_at=row.resolved_at,
    )


class SqlApprovalGate:
    """:class:`~src.agents.base.ApprovalGate` 的 SQLAlchemy 实现。

    每个方法管理自己的事务（审批操作独立于业务事务边界）。
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        """初始化网关。

        Args:
            session_factory: 数据库会话工厂。
        """
        self._session_factory = session_factory

    async def create(
        self,
        *,
        run_id: UUID,
        tenant_id: UUID,
        tool_name: str,
        arguments: dict[str, Any],
        ttl_seconds: int,
    ) -> ApprovalRequest:
        """创建审批请求（幂等）。见 :meth:`src.agents.base.ApprovalGate.create`。"""
        approval_id = _deterministic_id(run_id, arguments)
        async with self._session_factory() as session:
            existing = await session.get(ApprovalModel, approval_id)
            if existing is not None:
                return _to_request(existing)
            row = ApprovalModel(
                id=approval_id,
                run_id=run_id,
                tenant_id=tenant_id,
                tool_name=tool_name,
                arguments=arguments,
                status="pending",
                expires_at=datetime.now(tz=UTC) + timedelta(seconds=ttl_seconds),
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError:
                # 并发重放竞争：另一执行已插入同 ID，读回即可
                await session.rollback()
                raced = await session.get(ApprovalModel, approval_id)
                if raced is None:  # pragma: no cover - 不可达防御
                    raise DatabaseError("approval insert race lost twice") from None
                return _to_request(raced)
            except SQLAlchemyError as exc:
                raise DatabaseError("approval insert failed", cause=exc) from exc
            await session.refresh(row)
            _logger.info(
                "approval_created",
                approval_id=str(approval_id),
                run_id=str(run_id),
                tool=tool_name,
            )
            return _to_request(row)

    async def resolve(
        self, approval_id: UUID, *, approved: bool, resolver: Principal
    ) -> ApprovalRequest:
        """决议审批。见 :meth:`src.agents.base.ApprovalGate.resolve`。"""
        async with self._session_factory() as session:
            row = await session.get(ApprovalModel, approval_id)
            if row is None or row.tenant_id != resolver.tenant_id:
                raise NotFoundError(
                    "approval not found", details={"approval_id": str(approval_id)}
                )
            if row.status != "pending":
                raise ConflictError(
                    f"approval already {row.status}",
                    details={"approval_id": str(approval_id)},
                )
            if row.expires_at <= _now_like(row.expires_at):
                raise ConflictError(
                    "approval expired", details={"approval_id": str(approval_id)}
                )
            row.status = "approved" if approved else "rejected"
            row.resolver = resolver.id
            row.resolved_at = datetime.now(tz=UTC)
            try:
                await session.commit()
            except SQLAlchemyError as exc:
                raise DatabaseError("approval update failed", cause=exc) from exc
            await session.refresh(row)
            _logger.info(
                "approval_resolved",
                approval_id=str(approval_id),
                approved=approved,
                resolver=resolver.id,
            )
            return _to_request(row)

    async def get(self, approval_id: UUID, *, tenant_id: UUID) -> ApprovalRequest:
        """按租户查询。见 :meth:`src.agents.base.ApprovalGate.get`。"""
        async with self._session_factory() as session:
            row = await session.get(ApprovalModel, approval_id)
            if row is None or row.tenant_id != tenant_id:
                raise NotFoundError(
                    "approval not found", details={"approval_id": str(approval_id)}
                )
            return _to_request(row)

    async def load(self, approval_id: UUID) -> ApprovalRequest:
        """内部加载。见 :meth:`src.agents.base.ApprovalGate.load`。"""
        async with self._session_factory() as session:
            row = await session.get(ApprovalModel, approval_id)
            if row is None:
                raise NotFoundError(
                    "approval not found", details={"approval_id": str(approval_id)}
                )
            return _to_request(row)

    async def expire_overdue(self) -> int:
        """过期清理。见 :meth:`src.agents.base.ApprovalGate.expire_overdue`。

        注：将对应运行置为失败由 worker 的定时清理任务负责（阶段 5），
        本方法只推进审批状态。
        """
        async with self._session_factory() as session:
            try:
                result = await session.execute(
                    update(ApprovalModel)
                    .where(
                        ApprovalModel.status == "pending",
                        ApprovalModel.expires_at < datetime.now(tz=UTC),
                    )
                    .values(status="expired")
                )
                await session.commit()
            except SQLAlchemyError as exc:
                raise DatabaseError("approval expiry sweep failed", cause=exc) from exc
            count = int(getattr(result, "rowcount", 0) or 0)
            if count:
                _logger.info("approvals_expired", count=count)
            return count

    async def pending_for_run(self, run_id: UUID) -> ApprovalRequest | None:
        """返回运行当前未决的审批（恢复入口校验用）；无则 None。"""
        async with self._session_factory() as session:
            row = await session.scalar(
                select(ApprovalModel).where(
                    ApprovalModel.run_id == run_id, ApprovalModel.status == "pending"
                )
            )
            return None if row is None else _to_request(row)
