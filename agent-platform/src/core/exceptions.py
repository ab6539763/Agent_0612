"""平台统一异常体系。

设计约定（见 docs/architecture.md 5.3）：

- 所有平台异常继承 :class:`AgentPlatformError`，携带稳定的机器可读 ``code``、
  人类可读 ``message``、结构化 ``details`` 与重试语义 ``retryable``。
- infrastructure 层负责把第三方异常（厂商 SDK、驱动）翻译为本体系，
  domain 与 api 层不得出现第三方异常类型。
- api 层全局异常处理器依据 ``http_status`` 映射为 RFC 9457 Problem Details；
  SSE 流中以 ``run_failed`` 事件承载。
"""

from __future__ import annotations

from typing import Any


class AgentPlatformError(Exception):
    """所有平台异常的基类。

    Attributes:
        code: 稳定的机器可读错误码（``snake_case``），客户端可据此分支处理。
        message: 人类可读错误描述，不得包含敏感信息（会出现在响应与日志中）。
        details: 结构化补充信息，输出前经脱敏 processor 处理。
        retryable: 调用方（含平台内重试装饰器）是否可以安全重试。
        http_status: API 层映射的 HTTP 状态码。
    """

    code: str = "internal_error"
    http_status: int = 500
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        """初始化异常。

        Args:
            message: 人类可读错误描述。
            details: 结构化补充信息。
            cause: 原始异常（保留链路，等价于 ``raise ... from cause``）。
        """
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details or {}
        if cause is not None:
            self.__cause__ = cause

    def to_problem(self) -> dict[str, Any]:
        """导出 RFC 9457 Problem Details 字典（不含 ``instance``/trace 字段）。

        Returns:
            可直接 JSON 序列化的 Problem Details 载荷。
        """
        return {
            "type": f"https://errors.agent-platform.dev/{self.code}",
            "title": self.code,
            "status": self.http_status,
            "detail": self.message,
            "retryable": self.retryable,
            **({"errors": self.details} if self.details else {}),
        }


# ---------------------------------------------------------------------------
# 通用错误
# ---------------------------------------------------------------------------


class ConfigurationError(AgentPlatformError):
    """配置缺失或非法（启动期 fail-fast，不应出现在请求路径）。"""

    code = "configuration_error"


class ValidationError(AgentPlatformError):
    """业务层输入校验失败（API 入参校验由 Pydantic/FastAPI 处理）。"""

    code = "validation_error"
    http_status = 422


class AuthenticationError(AgentPlatformError):
    """身份认证失败：API Key / JWT 缺失、无效或过期。"""

    code = "authentication_error"
    http_status = 401


class AuthorizationError(AgentPlatformError):
    """权限不足：主体缺少所需 scope 或工具权限。"""

    code = "authorization_error"
    http_status = 403


class NotFoundError(AgentPlatformError):
    """资源不存在或不属于当前租户。"""

    code = "not_found"
    http_status = 404


class ConflictError(AgentPlatformError):
    """资源状态冲突（重复创建、乐观锁失败、非法状态迁移）。"""

    code = "conflict"
    http_status = 409


class RateLimitedError(AgentPlatformError):
    """平台级限流触发。"""

    code = "rate_limited"
    http_status = 429
    retryable = True


class PromptInjectionDetectedError(AgentPlatformError):
    """输入被注入检测器判定为高危并拦截。"""

    code = "prompt_injection_detected"
    http_status = 400


# ---------------------------------------------------------------------------
# LLM 调用错误
# ---------------------------------------------------------------------------


class LLMError(AgentPlatformError):
    """LLM 调用错误基类。各 Provider 必须把厂商异常翻译为本类的子类。"""

    code = "llm_error"
    http_status = 502


class LLMTimeoutError(LLMError):
    """LLM 调用超时（可重试）。"""

    code = "llm_timeout"
    retryable = True


class LLMRateLimitError(LLMError):
    """上游模型限流 / 配额耗尽（可重试，遵循退避）。"""

    code = "llm_rate_limited"
    http_status = 429
    retryable = True


class LLMServerError(LLMError):
    """上游 5xx / 连接错误（可重试）。"""

    code = "llm_server_error"
    retryable = True


class LLMAuthenticationError(LLMError):
    """上游凭证无效（不可重试，需运维介入）。"""

    code = "llm_authentication_error"


class LLMContentFilterError(LLMError):
    """内容被上游安全策略拦截（不可重试）。"""

    code = "llm_content_filtered"
    http_status = 400


class LLMContextLengthError(LLMError):
    """请求超出模型上下文窗口（不可重试，应触发记忆压缩或截断）。"""

    code = "llm_context_length_exceeded"
    http_status = 400


class LLMInvalidResponseError(LLMError):
    """上游返回无法解析的载荷（流式 chunk 异常、JSON 损坏等）。"""

    code = "llm_invalid_response"
    retryable = True


class CircuitOpenError(LLMError):
    """熔断器开启，调用被快速失败（可在熔断恢复后重试）。"""

    code = "llm_circuit_open"
    http_status = 503
    retryable = True


# ---------------------------------------------------------------------------
# 工具系统错误
# ---------------------------------------------------------------------------


class ToolError(AgentPlatformError):
    """工具系统错误基类。"""

    code = "tool_error"


class ToolNotFoundError(ToolError):
    """请求的工具未注册或对当前主体不可见。"""

    code = "tool_not_found"
    http_status = 404


class ToolPermissionDeniedError(ToolError):
    """主体缺少工具声明的所需权限。"""

    code = "tool_permission_denied"
    http_status = 403


class ToolArgumentError(ToolError):
    """LLM 产出的工具参数未通过 args_schema 校验。

    注意：该错误默认不终止运行——执行器会把错误信息作为观察结果回喂给
    模型，让其修正参数重试。
    """

    code = "tool_argument_error"
    http_status = 422


class ToolExecutionError(ToolError):
    """工具执行期间抛出的业务错误。"""

    code = "tool_execution_error"


class ToolTimeoutError(ToolError):
    """工具执行超时（被执行器取消）。"""

    code = "tool_timeout"
    retryable = True


# ---------------------------------------------------------------------------
# 记忆与检索错误
# ---------------------------------------------------------------------------


class MemoryStoreError(AgentPlatformError):
    """记忆读写失败（Redis / pgvector 后端错误的统一翻译）。"""

    code = "memory_store_error"
    retryable = True


class MemoryCompressionError(AgentPlatformError):
    """记忆压缩任务失败（可安全重试，窗口暂时偏长）。"""

    code = "memory_compression_error"
    retryable = True


class DocumentParseError(AgentPlatformError):
    """文档解析失败（格式损坏、不支持的类型）。"""

    code = "document_parse_error"
    http_status = 422


class RetrievalError(AgentPlatformError):
    """检索管道执行失败。"""

    code = "retrieval_error"
    retryable = True


# ---------------------------------------------------------------------------
# 编排与任务错误
# ---------------------------------------------------------------------------


class GraphExecutionError(AgentPlatformError):
    """LangGraph 图执行失败（节点抛出未归类异常时的包装）。"""

    code = "graph_execution_error"


class GuardrailViolationError(AgentPlatformError):
    """运行触达护栏（最大迭代次数 / token 预算）被强制终止。"""

    code = "guardrail_violation"


class ApprovalPendingError(AgentPlatformError):
    """操作因等待人工审批而暂停（非失败态，调用方应转入审批流程）。"""

    code = "approval_pending"
    http_status = 202


class ApprovalRejectedError(AgentPlatformError):
    """人工审批被拒绝，运行终止。"""

    code = "approval_rejected"


class RunCancelledError(AgentPlatformError):
    """运行被调用方主动取消。"""

    code = "run_cancelled"


class CheckpointError(AgentPlatformError):
    """checkpoint 读写或恢复失败。"""

    code = "checkpoint_error"
    retryable = True


# ---------------------------------------------------------------------------
# 基础设施错误
# ---------------------------------------------------------------------------


class InfrastructureError(AgentPlatformError):
    """基础设施错误基类（驱动/客户端异常的统一翻译）。"""

    code = "infrastructure_error"
    retryable = True


class DatabaseError(InfrastructureError):
    """PostgreSQL 访问错误。"""

    code = "database_error"


class CacheError(InfrastructureError):
    """Redis 访问错误。"""

    code = "cache_error"


class QueueError(InfrastructureError):
    """任务队列投递/消费错误。"""

    code = "queue_error"


class LockAcquisitionError(InfrastructureError):
    """分布式锁获取失败（超时或被占用）。"""

    code = "lock_acquisition_error"
