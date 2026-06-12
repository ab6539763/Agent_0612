"""跨模块共享的核心领域类型。

只放真正被多个模块依赖的值对象（主体、Token 用量等），避免演变成杂物箱。
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ToolPermission(StrEnum):
    """工具权限枚举。

    工具声明其所需权限，:class:`Principal` 携带被授予的权限集，
    ToolRegistry 在列举与执行时做交集校验。
    """

    READ = "read"
    """只读类操作（检索、查询）。"""

    WRITE = "write"
    """产生持久化副作用的操作（写库、发消息）。"""

    NETWORK = "network"
    """访问外部网络（HTTP 请求、网页抓取）。"""

    CODE_EXEC = "code_exec"
    """执行动态代码（沙箱内）。"""

    ADMIN = "admin"
    """管理类操作（动态注册工具、变更配置）。"""


class PrincipalType(StrEnum):
    """调用主体类型。"""

    API_KEY = "api_key"
    USER = "user"
    SERVICE = "service"


class Principal(BaseModel):
    """经鉴权解析后的调用主体，贯穿请求上下文与审计日志。

    由 API 层的鉴权依赖（API Key 或 JWT）构造，domain 层只消费不构造。
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="主体唯一标识（用户 ID / API Key ID）。")
    type: PrincipalType = Field(description="主体类型。")
    tenant_id: UUID = Field(description="所属租户，所有数据访问按此隔离。")
    permissions: frozenset[ToolPermission] = Field(
        default=frozenset(),
        description="被授予的工具权限集合。",
    )
    scopes: frozenset[str] = Field(
        default=frozenset(),
        description="API 级 scope（如 'runs:write', 'documents:read'）。",
    )


class TokenUsage(BaseModel):
    """一次或多次 LLM 调用的 token 用量统计，支持累加聚合。"""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        """提示与补全 token 之和。"""
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """合并两次用量统计（用于跨节点/跨调用聚合）。"""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class AuditStamp(BaseModel):
    """实体通用审计字段（由仓储层填充）。"""

    model_config = ConfigDict(frozen=True)

    created_at: datetime
    updated_at: datetime
