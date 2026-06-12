"""SqlAgentRunRepository 测试（AsyncSession Mock；真库集成在阶段 5 compose 环境）。"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions import NotFoundError
from src.infrastructure.models import AgentRunModel
from src.infrastructure.repositories import SqlAgentRunRepository


def make_session(**overrides: Any) -> MagicMock:
    session = MagicMock(spec=AsyncSession)
    session.flush = AsyncMock()
    session.execute = AsyncMock(**overrides.get("execute", {}))
    session.scalar = AsyncMock(**overrides.get("scalar", {}))
    return session


class TestCreate:
    async def test_adds_model_and_flushes(self) -> None:
        session = make_session()
        repo = SqlAgentRunRepository(cast(AsyncSession, session))
        run_id, tenant_id = uuid4(), uuid4()

        await repo.create(
            run_id=run_id,
            tenant_id=tenant_id,
            payload={"agent": "react", "model": "openai:gpt-4o", "input": "hi"},
        )

        added = session.add.call_args[0][0]
        assert isinstance(added, AgentRunModel)
        assert added.id == run_id
        assert added.status == "queued"
        assert added.model == "openai:gpt-4o"
        session.flush.assert_awaited_once()


class TestTransition:
    async def test_cas_success(self) -> None:
        session = make_session(execute={"return_value": SimpleNamespace(rowcount=1)})
        repo = SqlAgentRunRepository(cast(AsyncSession, session))
        assert await repo.transition(uuid4(), from_status="queued", to_status="running")

    async def test_cas_conflict_returns_false(self) -> None:
        session = make_session(execute={"return_value": SimpleNamespace(rowcount=0)})
        repo = SqlAgentRunRepository(cast(AsyncSession, session))
        assert not await repo.transition(
            uuid4(), from_status="queued", to_status="running"
        )


class TestGet:
    async def test_found(self) -> None:
        row = AgentRunModel(
            id=uuid4(),
            tenant_id=uuid4(),
            conversation_id=None,
            agent="react",
            model="openai:gpt-4o",
            status="running",
            input="hi",
            final_answer=None,
            error=None,
            prompt_tokens=10,
            completion_tokens=2,
            trace_id=None,
        )
        row.created_at = row.updated_at = datetime.now(tz=UTC)
        session = make_session(scalar={"return_value": row})
        repo = SqlAgentRunRepository(cast(AsyncSession, session))

        record = await repo.get(row.id, tenant_id=row.tenant_id)

        assert record["status"] == "running"
        assert record["prompt_tokens"] == 10

    async def test_missing_raises_not_found(self) -> None:
        session = make_session(scalar={"return_value": None})
        repo = SqlAgentRunRepository(cast(AsyncSession, session))
        with pytest.raises(NotFoundError):
            await repo.get(uuid4(), tenant_id=uuid4())
