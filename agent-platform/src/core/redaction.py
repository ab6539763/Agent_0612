"""敏感信息脱敏。

被 logging（日志 processor）、agents（事件下发前）、tools（审计）共用。
两类策略：

- **键名脱敏**：字典中键名命中敏感模式（password / api_key / token ...）时
  整值替换为 ``[REDACTED]``。
- **内容脱敏**：文本中命中正则（Bearer 凭证、sk- 密钥、邮箱、手机号、卡号）
  的片段替换为打码占位。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_SENSITIVE_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:password|passwd|secret|token|api_?key|authorization|credential|"
    r"private_?key|cookie|session_?id)(?:$|_)",
    re.IGNORECASE,
)

_CONTENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Bearer / Basic 凭证
    (re.compile(r"(?i)\b(bearer|basic)\s+[a-z0-9._~+/=-]{8,}"), r"\1 [REDACTED]"),
    # OpenAI / Anthropic 风格密钥
    (re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_-]{16,}\b"), REDACTED),
    # 平台 API Key（security.generate_api_key 的格式）
    (re.compile(r"\bap_[A-Za-z0-9_-]{16,}\b"), REDACTED),
    # JWT（三段 base64url）
    (
        re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
        REDACTED,
    ),
    # 邮箱：保留首字符与域名
    (
        re.compile(r"\b([A-Za-z0-9])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b"),
        r"\1***@\2",
    ),
    # 中国大陆手机号：保留前三后四
    (re.compile(r"\b(1[3-9]\d)\d{4}(\d{4})\b"), r"\1****\2"),
    # 16-19 位连续数字（银行卡号）
    (re.compile(r"\b\d{16,19}\b"), REDACTED),
)


def is_sensitive_key(key: str) -> bool:
    """判断字典键名是否属于敏感字段。"""
    return _SENSITIVE_KEY_PATTERN.search(key) is not None


def redact_text(text: str) -> str:
    """对文本做内容脱敏。

    Args:
        text: 原始文本。

    Returns:
        命中模式的片段被打码后的文本。
    """
    for pattern, replacement in _CONTENT_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def redact_value(value: Any, *, _depth: int = 0) -> Any:
    """递归脱敏任意 JSON 形值（dict / list / str 原样递归，其余类型透传）。

    Args:
        value: 待脱敏的值。

    Returns:
        脱敏后的副本（不修改原对象）。
    """
    if _depth > 16:  # 防御恶意深嵌套
        return REDACTED
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, Mapping):
        return {
            key: REDACTED
            if isinstance(key, str) and is_sensitive_key(key)
            else redact_value(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        items = [redact_value(item, _depth=_depth + 1) for item in value]
        return items if isinstance(value, list) else tuple(items)
    return value
