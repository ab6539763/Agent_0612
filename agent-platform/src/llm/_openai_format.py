"""OpenAI Chat Completions 格式的请求构建与响应解析。

被 :class:`~src.llm.openai_provider.OpenAIProvider` 与
:class:`~src.llm.litellm_provider.LiteLLMProvider` 共用——litellm 的响应对象
是 OpenAI 格式的鸭子类型，因此解析函数统一以 ``Any`` 接收并防御性取值。
"""

from __future__ import annotations

import json
from typing import Any

from src.core.exceptions import LLMInvalidResponseError
from src.core.types import TokenUsage
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ToolCallRequest,
)

_FINISH_REASONS: dict[str, FinishReason] = {
    "stop": FinishReason.STOP,
    "length": FinishReason.LENGTH,
    "tool_calls": FinishReason.TOOL_CALLS,
    "function_call": FinishReason.TOOL_CALLS,
    "content_filter": FinishReason.CONTENT_FILTER,
}


def strip_model_prefix(model: str) -> str:
    """去掉平台路由前缀（``openai:gpt-4o`` → ``gpt-4o``）。"""
    return model.split(":", 1)[1] if ":" in model else model


def map_finish_reason(raw: str | None) -> FinishReason:
    """归一化 finish_reason；未知值按 STOP 处理（防上游新增枚举破坏解析）。"""
    if raw is None:
        return FinishReason.STOP
    return _FINISH_REASONS.get(raw, FinishReason.STOP)


def messages_to_openai(messages: tuple[ChatMessage, ...]) -> list[dict[str, Any]]:
    """把平台消息转为 OpenAI wire format。"""
    payload: list[dict[str, Any]] = []
    for message in messages:
        if message.role is ChatRole.TOOL:
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.tool_call_id,
                    "content": message.content,
                }
            )
            continue
        item: dict[str, Any] = {"role": message.role.value, "content": message.content}
        if message.tool_calls:
            item["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in message.tool_calls
            ]
            if not message.content:
                item["content"] = None
        payload.append(item)
    return payload


def build_request_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """构建 ``chat.completions.create`` 的公共参数。"""
    kwargs: dict[str, Any] = {
        "model": strip_model_prefix(request.model),
        "messages": messages_to_openai(request.messages),
        "temperature": request.temperature,
    }
    if request.max_tokens is not None:
        kwargs["max_tokens"] = request.max_tokens
    if request.stop:
        kwargs["stop"] = list(request.stop)
    if request.tools:
        kwargs["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in request.tools
        ]
        kwargs["tool_choice"] = request.tool_choice
    return kwargs


def parse_arguments(raw_arguments: str | None) -> dict[str, Any]:
    """解析工具调用参数 JSON。

    Raises:
        LLMInvalidResponseError: 参数不是合法 JSON object。
    """
    if not raw_arguments:
        return {}
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            "tool call arguments are not valid JSON",
            details={"snippet": raw_arguments[:200]},
            cause=exc,
        ) from exc
    if not isinstance(parsed, dict):
        raise LLMInvalidResponseError(
            "tool call arguments must be a JSON object",
            details={"snippet": raw_arguments[:200]},
        )
    return parsed


def parse_usage(usage: Any) -> TokenUsage:
    """解析 usage 对象（缺失字段按 0 处理）。"""
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )


def parse_completion(response: Any, *, requested_model: str) -> CompletionResult:
    """解析非流式响应为平台结果。

    Args:
        response: OpenAI 格式的 ChatCompletion（或 litellm ModelResponse）。
        requested_model: 平台侧请求的模型名（含前缀，回填到结果）。

    Raises:
        LLMInvalidResponseError: 响应缺少 choices 或结构异常。
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise LLMInvalidResponseError("response has no choices")
    choice = choices[0]
    raw_message = choice.message

    tool_calls: list[ToolCallRequest] = []
    for raw_call in getattr(raw_message, "tool_calls", None) or []:
        tool_calls.append(
            ToolCallRequest(
                id=raw_call.id,
                name=raw_call.function.name,
                arguments=parse_arguments(raw_call.function.arguments),
            )
        )

    message = ChatMessage(
        role=ChatRole.ASSISTANT,
        content=getattr(raw_message, "content", None) or "",
        tool_calls=tuple(tool_calls),
    )
    return CompletionResult(
        message=message,
        finish_reason=map_finish_reason(getattr(choice, "finish_reason", None)),
        usage=parse_usage(getattr(response, "usage", None)),
        model=requested_model,
    )


class StreamingToolCallAggregator:
    """聚合流式响应中按 index 分片下发的工具调用增量。

    OpenAI 流式协议把一个工具调用拆为多个 delta（首个含 id/name，后续含
    参数 JSON 片段），本类按 index 聚合，流结束后一次性解析。
    """

    def __init__(self) -> None:
        """初始化空聚合器。"""
        self._calls: dict[int, dict[str, Any]] = {}

    def feed(self, delta_tool_calls: Any) -> None:
        """喂入一个 chunk 的 ``delta.tool_calls`` 列表（可为 None）。"""
        for raw in delta_tool_calls or []:
            index = getattr(raw, "index", 0) or 0
            slot = self._calls.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            if getattr(raw, "id", None):
                slot["id"] = raw.id
            function = getattr(raw, "function", None)
            if function is not None:
                if getattr(function, "name", None):
                    slot["name"] += function.name
                if getattr(function, "arguments", None):
                    slot["arguments"] += function.arguments

    def finalize(self) -> list[ToolCallRequest]:
        """解析聚合完成的工具调用（按 index 升序）。

        Raises:
            LLMInvalidResponseError: 缺少 id/name 或参数 JSON 非法。
        """
        results: list[ToolCallRequest] = []
        for index in sorted(self._calls):
            slot = self._calls[index]
            if not slot["id"] or not slot["name"]:
                raise LLMInvalidResponseError(
                    "streamed tool call missing id or name",
                    details={"index": index},
                )
            results.append(
                ToolCallRequest(
                    id=slot["id"],
                    name=slot["name"],
                    arguments=parse_arguments(slot["arguments"]),
                )
            )
        return results
