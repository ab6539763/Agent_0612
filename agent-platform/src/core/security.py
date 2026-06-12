"""安全组件：API Key、JWT、Prompt 注入检测。

鉴权数据流（API 层在阶段 5 接线）：

- ``X-API-Key`` → :func:`hash_api_key` → 查 ``api_keys`` 表 → ``Principal``。
- ``Authorization: Bearer <jwt>`` → :class:`JwtCodec.decode` → ``Principal``。

Prompt 注入防护采用规则启发式（ADR 见架构文档 5.1 节）：高风险输入直接拦截，
中风险打标降权（由 agents 层在系统提示中声明该内容不可信）。
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import time
from typing import Any
from uuid import UUID

import jwt as pyjwt
from pydantic import BaseModel, ConfigDict, Field

from src.core.config import SecuritySettings
from src.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    PromptInjectionDetectedError,
)
from src.core.types import Principal, PrincipalType, ToolPermission

# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------

API_KEY_PREFIX = "ap_"


def generate_api_key() -> str:
    """生成一个新的明文 API Key（仅在创建时返回一次，平台只存哈希）。

    Returns:
        形如 ``ap_<43 字符 urlsafe>`` 的密钥。
    """
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


def hash_api_key(raw_key: str, settings: SecuritySettings) -> str:
    """计算 API Key 的存储哈希（HMAC-SHA256，pepper 为密钥）。

    Args:
        raw_key: 明文 API Key。
        settings: 安全配置（提供 pepper）。

    Returns:
        十六进制哈希串，可直接作为 ``api_keys.key_hash`` 查询条件。
    """
    pepper = settings.api_key_pepper.get_secret_value().encode()
    return hmac.new(pepper, raw_key.encode(), hashlib.sha256).hexdigest()


def verify_api_key(raw_key: str, stored_hash: str, settings: SecuritySettings) -> bool:
    """常数时间比较 API Key 与存储哈希。"""
    return hmac.compare_digest(hash_api_key(raw_key, settings), stored_hash)


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


class JwtClaims(BaseModel):
    """平台 JWT 载荷的业务字段。"""

    model_config = ConfigDict(frozen=True)

    sub: str = Field(description="用户标识。")
    tenant_id: UUID
    scopes: tuple[str, ...] = ()
    permissions: tuple[ToolPermission, ...] = ()


class JwtCodec:
    """JWT 编解码器（HS256 开发 / RS256 生产）。"""

    def __init__(self, settings: SecuritySettings) -> None:
        """初始化编解码器。

        Args:
            settings: 安全配置。

        Raises:
            ConfigurationError: RS256 模式下缺少公钥。
        """
        self._settings = settings
        if settings.jwt_algorithm == "RS256" and not settings.jwt_public_key:
            raise ConfigurationError("RS256 requires SECURITY__JWT_PUBLIC_KEY")

    @property
    def _encode_key(self) -> str:
        if self._settings.jwt_algorithm == "HS256":
            return self._settings.jwt_secret.get_secret_value()
        if self._settings.jwt_private_key is None:
            raise ConfigurationError("RS256 signing requires SECURITY__JWT_PRIVATE_KEY")
        return self._settings.jwt_private_key.get_secret_value()

    @property
    def _decode_key(self) -> str:
        if self._settings.jwt_algorithm == "HS256":
            return self._settings.jwt_secret.get_secret_value()
        assert self._settings.jwt_public_key is not None
        return self._settings.jwt_public_key

    def encode(self, claims: JwtClaims, *, ttl_seconds: int | None = None) -> str:
        """签发 JWT。

        Args:
            claims: 业务载荷。
            ttl_seconds: 有效期；缺省用配置值。

        Returns:
            紧凑序列化的 JWT。
        """
        now = int(time.time())
        payload: dict[str, Any] = {
            "sub": claims.sub,
            "tenant_id": str(claims.tenant_id),
            "scopes": list(claims.scopes),
            "permissions": [p.value for p in claims.permissions],
            "iss": self._settings.jwt_issuer,
            "aud": self._settings.jwt_audience,
            "iat": now,
            "exp": now + (ttl_seconds or self._settings.jwt_ttl_seconds),
        }
        return pyjwt.encode(payload, self._encode_key, algorithm=self._settings.jwt_algorithm)

    def decode(self, token: str) -> Principal:
        """校验并解析 JWT 为调用主体。

        Args:
            token: 紧凑序列化的 JWT。

        Returns:
            解析出的 ``Principal``（type=USER）。

        Raises:
            AuthenticationError: 签名无效、过期、issuer/audience 不匹配或载荷缺字段。
        """
        try:
            payload = pyjwt.decode(
                token,
                self._decode_key,
                algorithms=[self._settings.jwt_algorithm],
                issuer=self._settings.jwt_issuer,
                audience=self._settings.jwt_audience,
                options={"require": ["exp", "iat", "sub"]},
            )
        except pyjwt.ExpiredSignatureError as exc:
            raise AuthenticationError("token expired", cause=exc) from exc
        except pyjwt.InvalidTokenError as exc:
            raise AuthenticationError("invalid token", cause=exc) from exc

        try:
            return Principal(
                id=str(payload["sub"]),
                type=PrincipalType.USER,
                tenant_id=UUID(payload["tenant_id"]),
                scopes=frozenset(str(s) for s in payload.get("scopes", [])),
                permissions=frozenset(
                    ToolPermission(p) for p in payload.get("permissions", [])
                ),
            )
        except (KeyError, ValueError) as exc:
            raise AuthenticationError("malformed token claims", cause=exc) from exc


# ---------------------------------------------------------------------------
# Prompt 注入检测
# ---------------------------------------------------------------------------


class InjectionVerdict(BaseModel):
    """注入检测结论。"""

    model_config = ConfigDict(frozen=True)

    risk: float = Field(ge=0.0, le=1.0, description="风险分（0 安全 ~ 1 高危）。")
    matched_rules: tuple[str, ...] = Field(default=())

    @property
    def is_suspicious(self) -> bool:
        """是否命中任意规则（中风险及以上，建议打标降权）。"""
        return self.risk > 0.0


class _Rule(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    pattern: re.Pattern[str]
    weight: float


_RULES: tuple[_Rule, ...] = (
    _Rule(
        name="override_instructions",
        pattern=re.compile(
            r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|prompts?|rules?)|忽略(之前|以上|上面|先前)的?(所有)?(指令|提示|规则|设定)"
        ),
        weight=0.6,
    ),
    _Rule(
        name="reveal_system_prompt",
        pattern=re.compile(
            r"(?i)(reveal|show|print|repeat|output)\s+(your\s+)?(system\s+prompt|instructions|"
            r"initial\s+prompt)|(输出|打印|重复|展示|告诉我)你的?(系统提示词?|初始指令|原始设定)"
        ),
        weight=0.5,
    ),
    _Rule(
        name="role_hijack",
        pattern=re.compile(
            r"(?i)you\s+are\s+now\s+(?!going)|pretend\s+(to\s+be|you\s+are)|act\s+as\s+if\s+you|"
            r"现在你(是|扮演)|假装你(是|没有)|从现在开始你"
        ),
        weight=0.4,
    ),
    _Rule(
        name="fake_message_boundary",
        pattern=re.compile(
            r"(?i)<\s*/?\s*(system|assistant)\s*>|\[/?(?:SYSTEM|INST)\]|"
            r"^\s*(system|assistant)\s*:",
            re.MULTILINE,
        ),
        weight=0.5,
    ),
    _Rule(
        name="exfiltrate_secrets",
        pattern=re.compile(
            r"(?i)(api[_\s-]?keys?|passwords?|credentials?|secrets?|环境变量|密钥|凭证)"
            r".{0,32}(send|post|upload|leak|发送|上传|泄露|提交)|"
            r"(send|post|upload|发送|上传).{0,32}(api[_\s-]?keys?|secrets?|密钥|凭证)"
        ),
        weight=0.6,
    ),
    _Rule(
        name="tool_abuse",
        pattern=re.compile(
            r"(?i)(call|invoke|use)\s+the\s+\w+\s+tool\s+with(out)?\s+(approval|asking|"
            r"confirmation)|不要(询问|确认|审批).{0,16}(直接)?(调用|执行)工具"
        ),
        weight=0.5,
    ),
)


class PromptInjectionDetector:
    """规则启发式 Prompt 注入检测器。

    设计取舍：规则检测召回有限但零延迟、可解释、无误杀大盘；分类器模型可在
    后续作为第二级接入（实现同一 ``inspect`` 签名替换）。
    """

    def __init__(self, settings: SecuritySettings) -> None:
        """初始化检测器。

        Args:
            settings: 安全配置（提供拦截阈值）。
        """
        self._block_threshold = settings.injection_block_threshold

    def inspect(self, text: str) -> InjectionVerdict:
        """检测文本的注入风险。

        Args:
            text: 待检测的不可信输入（用户消息、工具返回的外部内容）。

        Returns:
            风险结论；多条规则命中时风险按补集叠加（``1-Π(1-w)``）。
        """
        matched: list[str] = []
        safe_probability = 1.0
        for rule in _RULES:
            if rule.pattern.search(text):
                matched.append(rule.name)
                safe_probability *= 1.0 - rule.weight
        risk = round(1.0 - safe_probability, 4)
        return InjectionVerdict(risk=risk, matched_rules=tuple(matched))

    def ensure_allowed(self, text: str) -> InjectionVerdict:
        """检测并在超过拦截阈值时抛错。

        Args:
            text: 待检测输入。

        Returns:
            未达拦截阈值时返回结论（调用方可据 ``is_suspicious`` 打标）。

        Raises:
            PromptInjectionDetectedError: 风险分达到拦截阈值。
        """
        verdict = self.inspect(text)
        if verdict.risk >= self._block_threshold:
            raise PromptInjectionDetectedError(
                "input blocked by prompt-injection policy",
                details={"risk": verdict.risk, "rules": list(verdict.matched_rules)},
            )
        return verdict
