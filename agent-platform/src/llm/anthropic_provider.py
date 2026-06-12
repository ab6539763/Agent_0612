"""Anthropic 官方 async SDK 适配器。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anthropic
from anthropic import AsyncAnthropic

from src.core.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from src.core.types import TokenUsage
from src.llm._instrumentation import observe_llm_call
from src.llm._openai_format import strip_model_prefix
from src.llm.base import (
    ChatMessage,
    ChatRole,
    CompletionChunk,
    CompletionRequest,
    CompletionResult,
    FinishReason,
    ToolCallRequest,
)

_PROVIDER = "anthropic"

_STOP_REASONS: dict[str, FinishReason] = {
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.LENGTH,
    "tool_use": FinishReason.TOOL_CALLS,
}

_DEFAULT_MAX_TOKENS = 4096
"""Anthropic API 强制要求 max_tokens；请求未指定时的兜底值。"""


def _translate_error(exc: Exception) -> LLMError:
    """把 anthropic SDK 异常翻译为平台异常。"""
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeoutError("anthropic request timed out", cause=exc)
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimitError("anthropic rate limited", cause=exc)
    if isinstance(exc, anthropic.AuthenticationError | anthropic.PermissionDeniedError):
        return LLMAuthenticationError("anthropic credentials rejected", cause=exc)
    if isinstance(exc, anthropic.BadRequestError):
        text = str(exc).lower()
        if "prompt is too long" in text or "context" in text:
            return LLMContextLengthError("request exceeds model context window", cause=exc)
        return LLMError(f"anthropic rejected request: {exc}", cause=exc)
    if isinstance(exc, anthropic.APIStatusError) and exc.status_code >= 500:
        return LLMServerError(f"anthropic server error ({exc.status_code})", cause=exc)
    if isinstance(exc, anthropic.APIConnectionError):
        return LLMServerError("anthropic connection error", cause=exc)
    return LLMError(f"anthropic call failed: {exc}", cause=exc)


def _split_system(
    messages: tuple[ChatMessage, ...],
) -> tuple[str, list[dict[str, Any]]]:
    """拆出 system 文本并把其余消息转为 Anthropic wire format。

    Anthropic 的 system 是独立参数；tool 结果消息转为 user 角色的
    ``tool_result`` 内容块。
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.role is ChatRole.SYSTEM:
            system_parts.append(message.content)
            continue
        if message.role is ChatRole.TOOL:
            converted.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                }
            )
            continue
        if message.role is ChatRole.ASSISTANT and message.tool_calls:
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
                for call in message.tool_calls
            )
            converted.append({"role": "assistant", "content": blocks})
            continue
        converted.append({"role": message.role.value, "content": message.content})
    return "\n\n".join(system_parts), converted


def _build_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """构建 ``messages.create`` 参数。"""
    system, messages = _split_system(request.messages)
    kwargs: dict[str, Any] = {
        "model": strip_model_prefix(request.model),
        "messages": messages,
        "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
        "temperature": min(request.temperature, 1.0),  # anthropic 上限 1.0
    }
    if system:
        kwargs["system"] = system
    if request.stop:
        kwargs["stop_sequences"] = list(request.stop)
    if request.tools and request.tool_choice != "none":
        kwargs["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in request.tools
        ]
        kwargs["tool_choice"] = (
            {"type": "any"} if request.tool_choice == "required" else {"type": "auto"}
        )
    return kwargs


class AnthropicProvider:
    """:class:`~src.llm.base.LLMProvider` 的 Anthropic 实现。"""

    def __init__(
        self, *, api_key: str, client: AsyncAnthropic | None = None
    ) -> None:
        """初始化 Provider。

        Args:
            api_key: Anthropic API Key。
            client: 注入现成客户端（测试用）。
        """
        self._client = client or AsyncAnthropic(api_key=api_key)

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """非流式补全。见 :meth:`src.llm.base.LLMProvider.complete`。"""
        kwargs = _build_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=False
        ) as observation:
            try:
                response = await self._client.messages.create(
                    **kwargs, timeout=request.timeout_seconds
                )
            except Exception as exc:
                raise _translate_error(exc) from exc

            text_parts: list[str] = []
            tool_calls: list[ToolCallRequest] = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    if not isinstance(block.input, dict):
                        raise LLMInvalidResponseError(
                            "tool_use input is not an object",
                            details={"tool": block.name},
                        )
                    tool_calls.append(
                        ToolCallRequest(id=block.id, name=block.name, arguments=block.input)
                    )
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
            )
            observation.usage = usage
            return CompletionResult(
                message=ChatMessage(
                    role=ChatRole.ASSISTANT,
                    content="".join(text_parts),
                    tool_calls=tuple(tool_calls),
                ),
                finish_reason=_STOP_REASONS.get(
                    response.stop_reason or "end_turn", FinishReason.STOP
                ),
                usage=usage,
                model=request.model,
            )

    async def stream(self, request: CompletionRequest) -> AsyncIterator[CompletionChunk]:
        """流式补全。见 :meth:`src.llm.base.LLMProvider.stream`。

        ``input_json_delta`` 片段在本方法内聚合，``content_block_stop`` 时
        解析为完整 ``ToolCallRequest`` 下发。
        """
        kwargs = _build_kwargs(request)
        async with observe_llm_call(
            provider=_PROVIDER, model=request.model, streaming=True
        ) as observation:
            # index → 进行中的 tool_use 块
            pending_tools: dict[int, dict[str, str]] = {}
            prompt_tokens = 0
            completion_tokens = 0
            finish_reason = FinishReason.STOP
            try:
                event_stream = await self._client.messages.create(
                    **kwargs, timeout=request.timeout_seconds, stream=True
                )
                async for event in event_stream:
                    if event.type == "message_start":
                        prompt_tokens = event.message.usage.input_tokens
                    elif event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            pending_tools[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "json": "",
                            }
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield CompletionChunk(content_delta=delta.text)
                        elif delta.type == "input_json_delta":
                            slot = pending_tools.get(event.index)
                            if slot is not None:
                                slot["json"] += delta.partial_json
                    elif event.type == "content_block_stop":
                        slot = pending_tools.pop(event.index, None)
                        if slot is not None:
                            yield CompletionChunk(
                                tool_call=_parse_tool_use(slot)
                            )
                    elif event.type == "message_delta":
                        if event.delta.stop_reason:
                            finish_reason = _STOP_REASONS.get(
                                event.delta.stop_reason, FinishReason.STOP
                            )
                        completion_tokens = event.usage.output_tokens
            except LLMError:
                raise
            except Exception as exc:
                raise _translate_error(exc) from exc

            usage = TokenUsage(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            observation.usage = usage
            yield CompletionChunk(finish_reason=finish_reason, usage=usage)


def _parse_tool_use(slot: dict[str, str]) -> ToolCallRequest:
    """解析聚合完成的 tool_use 块。

    Raises:
        LLMInvalidResponseError: 参数 JSON 非法。
    """
    raw = slot["json"] or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMInvalidResponseError(
            "tool_use input is not valid JSON",
            details={"tool": slot["name"], "snippet": raw[:200]},
            cause=exc,
        ) from exc
    if not isinstance(arguments, dict):
        raise LLMInvalidResponseError(
            "tool_use input must be a JSON object", details={"tool": slot["name"]}
        )
    return ToolCallRequest(id=slot["id"], name=slot["name"], arguments=arguments)
