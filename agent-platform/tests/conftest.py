"""测试共享 fixture。"""

from __future__ import annotations

from uuid import uuid4

import pytest
from src.core.config import (
    CircuitBreakerSettings,
    RetrySettings,
    SecuritySettings,
    Settings,
)
from src.core.types import Principal, PrincipalType, ToolPermission


@pytest.fixture()
def settings() -> Settings:
    """默认配置（不读 .env，全部用代码默认值）。"""
    return Settings(_env_file=None)


@pytest.fixture()
def security_settings() -> SecuritySettings:
    """安全配置（HS256）。"""
    return SecuritySettings()


@pytest.fixture()
def retry_settings() -> RetrySettings:
    """快速重试配置（测试无需真实等待，sleep 已注入为空操作）。"""
    return RetrySettings(max_attempts=3, base_delay_seconds=0.1, max_delay_seconds=1.0)


@pytest.fixture()
def breaker_settings() -> CircuitBreakerSettings:
    """小阈值熔断配置。"""
    return CircuitBreakerSettings(
        failure_threshold=3, recovery_seconds=10.0, half_open_max_probes=1
    )


@pytest.fixture()
def principal() -> Principal:
    """普通用户主体。"""
    return Principal(
        id="user-1",
        type=PrincipalType.USER,
        tenant_id=uuid4(),
        permissions=frozenset({ToolPermission.READ}),
        scopes=frozenset({"runs:write"}),
    )
