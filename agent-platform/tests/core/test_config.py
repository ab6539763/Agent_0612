"""config 模块测试。"""

from __future__ import annotations

import pytest
from src.core.config import Environment, Settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings(_env_file=None)
        assert settings.environment is Environment.DEVELOPMENT
        assert settings.database.pool_size == 10
        assert settings.llm.retry.max_attempts == 3
        assert not settings.is_production

    def test_nested_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DATABASE__POOL_SIZE", "42")
        monkeypatch.setenv("LLM__RETRY__MAX_ATTEMPTS", "5")
        monkeypatch.setenv("LLM__OPENAI_API_KEY", "sk-test-1234567890abcdef")

        settings = Settings(_env_file=None)

        assert settings.is_production
        assert settings.database.pool_size == 42
        assert settings.llm.retry.max_attempts == 5
        assert settings.llm.openai_api_key is not None

    def test_secret_not_in_repr(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM__OPENAI_API_KEY", "sk-super-secret-value-123")
        settings = Settings(_env_file=None)
        assert "sk-super-secret-value-123" not in repr(settings)
        assert "sk-super-secret-value-123" not in str(settings.llm)
