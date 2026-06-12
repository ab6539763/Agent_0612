"""Token 计数（tiktoken）。

用于记忆窗口预算与压缩触发判断。对非 OpenAI 模型是近似值——记忆预算场景
只需要量级正确，精确计费一律以 Provider 返回的 usage 为准。
"""

from __future__ import annotations

from functools import lru_cache

import tiktoken

from src.llm.base import ChatMessage

_MESSAGE_OVERHEAD_TOKENS = 4
"""每条消息的结构开销近似值（role 标记与分隔符）。"""


@lru_cache(maxsize=4)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class TokenCounter:
    """基于 tiktoken 的 token 计数器。"""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        """初始化计数器。

        Args:
            encoding_name: tiktoken 编码名；cl100k_base 对主流模型误差可接受。
        """
        self._encoding = _encoding(encoding_name)

    def count_text(self, text: str) -> int:
        """计算文本 token 数。"""
        return len(self._encoding.encode(text))

    def count_message(self, message: ChatMessage) -> int:
        """计算单条消息 token 数（含结构开销与工具调用参数）。"""
        total = _MESSAGE_OVERHEAD_TOKENS + self.count_text(message.content)
        for call in message.tool_calls:
            total += self.count_text(call.name)
            total += self.count_text(str(call.arguments))
        return total

    def count_messages(self, messages: tuple[ChatMessage, ...] | list[ChatMessage]) -> int:
        """计算消息序列总 token 数。"""
        return sum(self.count_message(m) for m in messages)
