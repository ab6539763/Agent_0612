"""SqlApprovalGate 测试（SQLite 内存库，模型为方言可移植类型）。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from src.agents.approval_gate import SqlApprovalGate
from src.core.exceptions import ConflictError, NotFoundError
from src.infrastructure.models import ApprovalModel, Base

from tests.agents.fakes import make_principal


@pytest.fixture()
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn, tables=[ApprovalModel.__table__]
            )
        )
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture()
def gate(session_factory: async_sessionmaker[AsyncSession]) -> SqlApprovalGate:
    return SqlApprovalGate(session_factory)


class TestCreate:
    async def test_create_pending(self, gate: SqlApprovalGate) -> None:
        run_id, tenant_id = uuid4(), uuid4()
        approval = await gate.create(
            run_id=run_id,
            tenant_id=tenant_id,
            tool_name="delete_everything",
            arguments={"calls": [{"id": "c1"}]},
            ttl_seconds=600,
        )
        assert approval.status == "pending"
        assert approval.run_id == run_id

    async def test_idempotent_on_replay(self, gate: SqlApprovalGate) -> None:
        run_id, tenant_id = uuid4(), uuid4()
        kwargs = {
            "run_id": run_id,
            "tenant_id": tenant_id,
            "tool_name": "t",
            "arguments": {"calls": [{"id": "c1", "name": "t"}]},
            "ttl_seconds": 600,
        }
        first = await gate.create(**kwargs)  # type: ignore[arg-type]
        second = await gate.create(**kwargs)  # type: ignore[arg-type]
        assert first.id == second.id  # 重放命中已有行

    async def test_different_arguments_different_id(self, gate: SqlApprovalGate) -> None:
        run_id, tenant_id = uuid4(), uuid4()
        a = await gate.create(
            run_id=run_id, tenant_id=tenant_id, tool_name="t",
            arguments={"calls": [{"id": "c1"}]}, ttl_seconds=600,
        )
        b = await gate.create(
            run_id=run_id, tenant_id=tenant_id, tool_name="t",
            arguments={"calls": [{"id": "c2"}]}, ttl_seconds=600,
        )
        assert a.id != b.id


class TestResolve:
    async def test_approve(self, gate: SqlApprovalGate) -> None:
        principal = make_principal()
        approval = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 1}, ttl_seconds=600,
        )
        resolved = await gate.resolve(approval.id, approved=True, resolver=principal)
        assert resolved.status == "approved"
        assert resolved.resolver == principal.id
        assert resolved.resolved_at is not None

    async def test_reject(self, gate: SqlApprovalGate) -> None:
        principal = make_principal()
        approval = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 2}, ttl_seconds=600,
        )
        resolved = await gate.resolve(approval.id, approved=False, resolver=principal)
        assert resolved.status == "rejected"

    async def test_double_resolve_conflicts(self, gate: SqlApprovalGate) -> None:
        principal = make_principal()
        approval = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 3}, ttl_seconds=600,
        )
        await gate.resolve(approval.id, approved=True, resolver=principal)
        with pytest.raises(ConflictError, match="already approved"):
            await gate.resolve(approval.id, approved=False, resolver=principal)

    async def test_cross_tenant_hidden(self, gate: SqlApprovalGate) -> None:
        approval = await gate.create(
            run_id=uuid4(), tenant_id=uuid4(), tool_name="t",
            arguments={"x": 4}, ttl_seconds=600,
        )
        outsider = make_principal()  # 不同租户
        with pytest.raises(NotFoundError):
            await gate.resolve(approval.id, approved=True, resolver=outsider)
        with pytest.raises(NotFoundError):
            await gate.get(approval.id, tenant_id=outsider.tenant_id)

    async def test_expired_cannot_resolve(self, gate: SqlApprovalGate) -> None:
        principal = make_principal()
        approval = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 5}, ttl_seconds=-10,  # 立即过期
        )
        with pytest.raises(ConflictError, match="expired"):
            await gate.resolve(approval.id, approved=True, resolver=principal)


class TestQueries:
    async def test_load_without_tenant(self, gate: SqlApprovalGate) -> None:
        approval = await gate.create(
            run_id=uuid4(), tenant_id=uuid4(), tool_name="t",
            arguments={"x": 6}, ttl_seconds=600,
        )
        loaded = await gate.load(approval.id)
        assert loaded.id == approval.id

    async def test_load_missing(self, gate: SqlApprovalGate) -> None:
        with pytest.raises(NotFoundError):
            await gate.load(uuid4())

    async def test_pending_for_run(self, gate: SqlApprovalGate) -> None:
        run_id = uuid4()
        assert await gate.pending_for_run(run_id) is None
        approval = await gate.create(
            run_id=run_id, tenant_id=uuid4(), tool_name="t",
            arguments={"x": 7}, ttl_seconds=600,
        )
        pending = await gate.pending_for_run(run_id)
        assert pending is not None
        assert pending.id == approval.id


class TestExpiry:
    async def test_expire_overdue_sweep(
        self,
        gate: SqlApprovalGate,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        principal = make_principal()
        stale = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 8}, ttl_seconds=600,
        )
        # 手动把 expires_at 拨到过去（绕过 create 的正向 TTL）
        async with session_factory() as session:
            row = await session.get(ApprovalModel, stale.id)
            assert row is not None
            row.expires_at = datetime.now(tz=UTC) - timedelta(minutes=5)
            await session.commit()
        fresh = await gate.create(
            run_id=uuid4(), tenant_id=principal.tenant_id, tool_name="t",
            arguments={"x": 9}, ttl_seconds=600,
        )

        expired_count = await gate.expire_overdue()

        assert expired_count == 1
        assert (await gate.load(stale.id)).status == "expired"
        assert (await gate.load(fresh.id)).status == "pending"
