"""平台配置：pydantic-settings 从环境变量加载。

环境变量命名：嵌套组用双下划线分隔，如 ``DATABASE__DSN`` / ``LLM__OPENAI_API_KEY``。
完整清单见仓库根目录 ``.env.example``。

约定：
- 配置在进程启动时加载一次（:func:`get_settings` 带缓存），运行期只读。
- 凭证类字段一律 ``SecretStr``，防止意外进入日志与 repr。
- 校验失败抛 ``ConfigurationError``，启动 fail-fast。
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.exceptions import ConfigurationError


class Environment(StrEnum):
    """运行环境。"""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class DatabaseSettings(BaseModel):
    """PostgreSQL 连接配置。"""

    dsn: str = Field(
        default="postgresql+asyncpg://agent:agent@localhost:5432/agent_platform",
        description="asyncpg DSN。",
    )
    pool_size: int = Field(default=10, gt=0)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout_seconds: float = Field(default=10.0, gt=0)
    pool_recycle_seconds: int = Field(default=1800, gt=0)
    echo: bool = Field(default=False, description="是否回显 SQL（仅开发）。")


class RedisSettings(BaseModel):
    """Redis 连接配置（缓存、会话、锁、事件流共用一个实例，按 key 前缀隔离）。"""

    url: str = Field(default="redis://localhost:6379/0")
    max_connections: int = Field(default=50, gt=0)
    socket_timeout_seconds: float = Field(default=5.0, gt=0)
    event_stream_ttl_seconds: int = Field(
        default=3600, gt=0, description="run 事件流的保留时间（断线续传窗口）。"
    )


class QueueSettings(BaseModel):
    """ARQ 任务队列配置。"""

    redis_url: str = Field(
        default="redis://localhost:6379/1",
        description="队列独立使用一个 Redis db，避免与缓存键互扰。",
    )
    queue_name: str = Field(default="agent_platform:queue")
    job_timeout_seconds: int = Field(default=1800, gt=0)
    max_tries: int = Field(default=3, gt=0)


class SecuritySettings(BaseModel):
    """鉴权与脱敏配置。"""

    api_key_pepper: SecretStr = Field(
        default=SecretStr("dev-pepper-change-me"),
        description="API Key 哈希的服务端 pepper（HMAC 密钥）。",
    )
    jwt_algorithm: Literal["HS256", "RS256"] = Field(default="HS256")
    jwt_secret: SecretStr = Field(
        default=SecretStr("dev-secret-change-me"),
        description="HS256 对称密钥；生产环境建议 RS256。",
    )
    jwt_private_key: SecretStr | None = Field(
        default=None, description="RS256 私钥（PEM），仅签发方需要。"
    )
    jwt_public_key: str | None = Field(
        default=None, description="RS256 公钥（PEM）。"
    )
    jwt_issuer: str = Field(default="agent-platform")
    jwt_audience: str = Field(default="agent-platform")
    jwt_ttl_seconds: int = Field(default=3600, gt=0)
    injection_block_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Prompt 注入风险分超过该阈值的输入直接拦截。",
    )


class RetrySettings(BaseModel):
    """LLM 调用重试策略（指数退避 + 全抖动）。"""

    max_attempts: int = Field(default=3, gt=0, le=10)
    base_delay_seconds: float = Field(default=0.5, gt=0)
    max_delay_seconds: float = Field(default=8.0, gt=0)


class CircuitBreakerSettings(BaseModel):
    """LLM 熔断器配置（按 Provider 维度独立）。"""

    failure_threshold: int = Field(
        default=5, gt=0, description="连续失败该次数后熔断开启。"
    )
    recovery_seconds: float = Field(
        default=30.0, gt=0, description="开启后经过该时长进入半开态放行探测。"
    )
    half_open_max_probes: int = Field(
        default=1, gt=0, description="半开态允许的并发探测请求数。"
    )


class LLMSettings(BaseModel):
    """模型接入配置。"""

    openai_api_key: SecretStr | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    anthropic_api_key: SecretStr | None = Field(default=None)
    vllm_base_url: str | None = Field(
        default=None, description="vLLM 的 OpenAI-compatible 端点，经 litellm 接入。"
    )
    vllm_api_key: SecretStr | None = Field(default=None)
    default_model: str = Field(default="openai:gpt-4o-mini")
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimension: int = Field(
        default=1536, gt=0, description="须与 pgvector 列维度一致（迁移时固化）。"
    )
    retry: RetrySettings = RetrySettings()
    circuit_breaker: CircuitBreakerSettings = CircuitBreakerSettings()


class ObservabilitySettings(BaseModel):
    """日志、追踪与指标配置。"""

    service_name: str = Field(default="agent-platform")
    log_level: str = Field(default="INFO")
    json_logs: bool = Field(default=True, description="False 时输出彩色控制台日志（开发）。")
    otlp_endpoint: str | None = Field(
        default=None, description="OTLP gRPC 端点；为空则不导出 trace。"
    )
    trace_sample_ratio: float = Field(default=1.0, ge=0.0, le=1.0)


class Settings(BaseSettings):
    """平台根配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    queue: QueueSettings = QueueSettings()
    security: SecuritySettings = SecuritySettings()
    llm: LLMSettings = LLMSettings()
    observability: ObservabilitySettings = ObservabilitySettings()

    @property
    def is_production(self) -> bool:
        """是否生产环境。"""
        return self.environment is Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """加载并缓存全局配置（进程级单例）。

    Returns:
        已校验的配置实例。

    Raises:
        ConfigurationError: 环境变量缺失或非法。
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(
            "settings validation failed",
            details={"errors": exc.errors(include_input=False, include_url=False)},
            cause=exc,
        ) from exc
