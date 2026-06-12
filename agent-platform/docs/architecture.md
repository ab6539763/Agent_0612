# Agent Platform 架构设计文档

> 版本：v0.1（阶段 1 产出）
> 状态：待评审
> 关联 ADR：见 `docs/adr/`

## 1. 系统概览

本平台是一个企业级 AI Agent 服务，对外提供「以 SSE 流式输出的多 Agent 任务执行
API」，对内提供可插拔的工具系统、双层记忆系统、RAG 检索管道与异步长任务调度。

```
                          ┌─────────────────────────────────────────────┐
                          │                  Clients                    │
                          │   (Web / SDK / 内部服务，SSE + REST)          │
                          └──────────────────────┬──────────────────────┘
                                                 │ HTTPS (API Key / JWT)
                          ┌──────────────────────▼──────────────────────┐
                          │              api（FastAPI）                  │
                          │  路由 / 中间件 / 鉴权 / 限流 / SSE 序列化       │
                          └──────────────────────┬──────────────────────┘
                                                 │ 仅依赖 service 接口
        ┌────────────────────────────────────────▼────────────────────────────────────┐
        │                            service / domain 层                               │
        │                                                                              │
        │  ┌───────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
        │  │  agents   │  │  tools   │  │  memory  │  │   rag    │  │     llm      │   │
        │  │ LangGraph │←→│ Registry │  │ 双层记忆  │  │ 混合检索  │  │  Provider    │   │
        │  │ 编排/状态  │  │ 权限控制  │  │ 压缩摘要  │  │ 重排序    │  │  抽象层      │   │
        │  └───────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
        │            （以上模块只依赖彼此的 Protocol / Pydantic 模型，不依赖具体实现）        │
        └────────────────────────────────────────┬────────────────────────────────────┘
                                                 │ 依赖倒置（接口在 domain，实现在下层）
        ┌────────────────────────────────────────▼────────────────────────────────────┐
        │                            infrastructure 层                                 │
        │   PostgreSQL(+pgvector) │ Redis(会话/锁/缓存) │ ARQ 任务队列 │ OTel/Prometheus  │
        └──────────────────────────────────────────────────────────────────────────────┘
```

## 2. 分层架构与依赖规则

| 层 | 目录 | 职责 | 允许依赖 |
|---|---|---|---|
| API 层 | `src/api` | HTTP 路由、请求/响应 Schema、中间件、SSE 序列化、依赖注入装配（组合根） | 下面所有层 |
| Service/Domain 层 | `src/agents` `src/tools` `src/memory` `src/rag` | 各模块的**接口与模型**（`base.py`，零基础设施依赖）+ 业务实现（可使用 infrastructure 客户端） | 彼此的 `base.py`、llm、infrastructure、core |
| 模型接入层 | `src/llm` | LLM/Embedding 统一抽象与 Provider 实现、重试/熔断 | core |
| 基础设施层 | `src/infrastructure` | 数据库引擎/会话、Redis 客户端、队列、事件流、仓储 | core |
| 核心层 | `src/core` | 配置、日志、异常体系、安全（鉴权/脱敏/注入防护）、共享类型 | 仅标准库与基础三方库 |

**强制规则**（由 import-linter contracts + ruff `flake8-tidy-imports` 在 CI 校验）：

1. 依赖单向流动：`api → (agents|tools|memory|rag) → llm → infrastructure → core`，
   禁止任何反向 import。
2. 各模块的 `base.py`（接口与模型）只允许依赖其他模块的 `base.py` 与 core，
   保持接口零基础设施依赖，单元测试可用纯内存 Fake 替换。
3. 跨模块协作只通过 Protocol；实例装配只发生在 `src/api/deps.py` 组合根与
   worker 入口，业务代码不直接实例化其他模块的具体实现。

## 3. 核心模块设计

### 3.1 llm — 模型适配层（ADR-0003）

- 统一接口 `LLMProvider`（`complete` / `stream`），输入输出为平台自有的
  `ChatMessage` / `CompletionChunk` 模型，与任何厂商 SDK 解耦。
- 内置三个实现（阶段 2 交付）：`OpenAIProvider`、`AnthropicProvider`、
  `LiteLLMProvider`（兜底覆盖 vLLM/OpenAI-compatible 等长尾模型）。
- 弹性策略在适配层之上以装饰器组合：`RetryingProvider`（指数退避 + 抖动，仅对
  可重试错误类别）→ `CircuitBreakerProvider`（半开探测）→ 具体 Provider。
- Embedding 单独抽象为 `EmbeddingProvider`，供 rag 与 memory 共用。
- 路由：`ProviderRouter` 按模型名前缀（`openai:gpt-4o` / `anthropic:claude-…` /
  `vllm:qwen-…`）选择 Provider，支持按租户配置降级链。

### 3.2 agents — LangGraph 编排（ADR-0002）

- 不使用已废弃的 LangChain `AgentExecutor`；所有编排基于 LangGraph `StateGraph`。
- 两种内置图模板：
  - **ReAct 图**：`reason → (tool_calls? → act → observe → reason) → respond`，
    带最大迭代数与 token 预算护栏。
  - **Planner-Executor 图**：`plan → route → executor(并行子任务) → review →
    (replan | respond)`，子任务执行节点可挂接多个专职 Agent（多 Agent 协作通过
    子图 + 共享状态通道实现）。
- 状态：`AgentState`（Pydantic 模型 + LangGraph reducer），可被 checkpointer
  序列化，支撑断点恢复与 Human-in-the-loop（`interrupt` 节点 → 审批 → resume）。
- 对外统一接口 `AgentRuntime.run_stream()`，产出 `AgentEvent` 异步流（见 3.6）。
- Checkpointer 使用 PostgreSQL（`langgraph-checkpoint-postgres`），与业务库同库
  不同 schema。

### 3.3 tools — 工具系统

- `BaseTool` ABC：声明 `name` / `description` / `args_schema`（Pydantic 模型，
  自动导出 JSON Schema 给 Function Calling）/ `required_permission` /
  `requires_approval`。
- `ToolRegistry`：进程内注册中心，支持启动时装饰器注册与运行时动态注册/注销；
  按调用方主体（`Principal`）的权限集过滤可见工具。
- 权限模型：`ToolPermission` 枚举（`read` / `write` / `network` / `code_exec` /
  `admin`），工具声明所需权限，API Key / JWT 的 scope 决定授予的权限集。
- 高危工具（`requires_approval=True`）触发 Human-in-the-loop：图执行暂停，
  发出 `approval_required` 事件，审批通过后从 checkpoint 恢复。
- 工具执行统一经 `ToolExecutor`：超时控制、参数校验、结果截断、审计日志。

### 3.4 memory — 双层记忆（ADR-0005）

- `ShortTermMemory`（Redis 实现）：会话级消息窗口，按 token 预算裁剪，TTL 管理。
- `LongTermMemory`（pgvector 实现）：跨会话语义记忆，`add / search / forget`，
  按 `tenant_id + user_id` 隔离。
- `MemoryCompressor`：当短期记忆超出预算时，由后台任务（ARQ）调用 LLM 生成
  渐进式摘要（rolling summary），摘要写回短期记忆头部、原始片段沉淀到长期记忆。
- 读路径：Agent 启动时组装 `system + summary + 长期记忆命中 + 窗口消息`。

### 3.5 rag — 检索管道（ADR-0006）

管道五阶段，每个阶段是独立 Protocol，可单独替换：

```
DocumentParser → Chunker → Embedder(=EmbeddingProvider) → Indexer
                                     查询侧：HybridRetriever(向量 + BM25, RRF 融合) → Reranker
```

- 解析：内置 PDF / Markdown / HTML / 纯文本解析器，统一产出 `Document`。
- 分块：递归字符分块为默认实现，保留标题路径等结构化元数据。
- 混合检索：pgvector（HNSW，余弦）+ PostgreSQL 全文检索（BM25 语义由
  `ts_rank` 近似，中文走 zhparser/jieba 分词），RRF（Reciprocal Rank Fusion）融合。
- 重排序：`Reranker` 接口，内置 cross-encoder（本地）与 LLM-as-reranker 两种实现。
- RAG 以普通工具（`knowledge_search`）形式注册进 ToolRegistry，由 Agent 决定何时检索。

### 3.6 流式事件协议（ADR-0007）

所有 Agent 输出统一为 `AgentEvent` 判别联合（`type` 字段判别）：

| 事件 | 含义 |
|---|---|
| `run_started` / `run_finished` / `run_failed` | 运行生命周期 |
| `plan_created` / `plan_updated` | Planner 产出/修订计划 |
| `step_started` / `step_finished` | 图节点级进度 |
| `reasoning_delta` | 推理过程增量文本（可按租户配置关闭） |
| `tool_call` / `tool_result` | 工具调用与结果（参数/结果经脱敏） |
| `message_delta` / `message_completed` | 最终回答的增量 token 与完整消息 |
| `approval_required` / `approval_resolved` | Human-in-the-loop |
| `heartbeat` | SSE 保活 |

API 层将事件序列化为 SSE（`event:` = 事件类型，`data:` = JSON，`id:` = 单调递增
序号用于断线重连 `Last-Event-ID` 续传，事件缓冲在 Redis Stream）。

### 3.7 任务队列与长任务（ADR-0004）

- 选型 **ARQ**（asyncio 原生）而非 Celery，理由见 ADR-0004。
- 长任务（深度研究、批量文档摄取、记忆压缩）投递到 ARQ；任务状态机：
  `queued → running → waiting_approval → running → succeeded | failed | cancelled`，
  持久化在 PostgreSQL `task_runs` 表。
- 断点恢复：worker 崩溃后任务重投，Agent 从 LangGraph checkpoint 续跑；
  任务级幂等键防止重复执行副作用。
- 审批：`waiting_approval` 状态由 `POST /v1/approvals/{id}` 推进，恢复执行。

## 4. 数据模型（核心表，阶段 2 落地为 SQLAlchemy 模型 + Alembic 迁移）

| 表 | 说明 |
|---|---|
| `tenants` / `api_keys` | 租户与 API Key（哈希存储），key 关联权限 scope |
| `conversations` / `messages` | 会话与消息（审计与回放） |
| `agent_runs` | 一次 Agent 执行：图类型、状态、token 用量、trace_id |
| `task_runs` | 异步任务状态机、幂等键、checkpoint 引用 |
| `approvals` | 审批请求：载荷快照、审批人、决策 |
| `documents` / `chunks` | RAG 文档与分块（`chunks.embedding vector(N)`，HNSW 索引；`tsv tsvector`，GIN 索引） |
| `memories` | 长期记忆条目（embedding + 元数据 + 衰减权重） |
| LangGraph checkpoint 表 | 独立 schema `checkpoints`，由官方 checkpointer 管理 |

## 5. 横切关注点

### 5.1 安全

- **鉴权**：服务间用 API Key（`X-API-Key`，数据库存 SHA-256 哈希）；终端用户用
  JWT（RS256，含 `tenant_id` / `scopes`）。两者统一解析为 `Principal` 注入请求上下文。
- **输入校验**：所有外部输入经 Pydantic v2 严格模式；上传文档限制类型与大小。
- **Prompt 注入防护**：① 系统提示与用户输入结构性隔离（独立消息角色，禁止字符串
  拼接进 system prompt）；② 工具结果包裹在定界标记中并声明为不可信内容；③ 注入
  启发式检测器（规则 + 可选分类器）对高危输入降权或拦截；④ 工具权限最小化，敏感
  工具强制审批。
- **脱敏日志**：structlog processor 对 `api_key` / `authorization` / email / 手机号
  等模式统一打码；LLM 请求体默认只记录摘要哈希，全文记录需显式开启且单独存储。

### 5.2 可观测性

- **日志**：structlog JSON 输出，绑定 `trace_id` / `tenant_id` / `run_id`。
- **追踪**：OpenTelemetry，FastAPI / SQLAlchemy / Redis / httpx 自动埋点 +
  LLM 调用、图节点、工具执行手动 span（含 token 用量属性）。
- **指标**：Prometheus —— 请求时延直方图、LLM 时延/Token/成本计数器、工具成功率、
  熔断器状态、队列深度、检索召回时延。
- **健康检查**：`/healthz`（存活）、`/readyz`（依赖就绪：PG/Redis/队列）。

### 5.3 错误处理

异常体系（完整定义见 `src/core/exceptions.py`）：

```
AgentPlatformError(基类, 含 code/message/details/retryable)
├── ConfigurationError
├── AuthenticationError / AuthorizationError
├── ValidationError
├── NotFoundError / ConflictError
├── RateLimitedError
├── LLMError
│   ├── LLMTimeoutError / LLMRateLimitError（可重试）
│   ├── LLMContentFilterError / LLMContextLengthError（不可重试）
│   └── CircuitOpenError
├── ToolError（ToolNotFoundError / ToolPermissionDeniedError / ToolExecutionError / ToolTimeoutError）
├── MemoryError* / RetrievalError
├── GraphExecutionError / ApprovalPendingError / RunCancelledError
└── InfrastructureError（DatabaseError / CacheError / QueueError / LockAcquisitionError）
```

API 层全局异常处理器把异常映射为 RFC 9457 Problem Details 响应；SSE 流中错误以
`run_failed` 事件下发后正常关闭连接。

## 6. 部署拓扑

- **开发**：Docker Compose —— `api` / `worker`（ARQ）/ `postgres(pgvector)` /
  `redis` / `otel-collector` / `prometheus` / `grafana`。
- **生产**：Kubernetes —— api Deployment（HPA，按 CPU + 进行中 SSE 连接数）、
  worker Deployment（按队列深度 KEDA 扩缩）、PG/Redis 建议托管服务；liveness =
  `/healthz`，readiness = `/readyz`；ConfigMap + Secret 注入环境变量。
- 镜像：多阶段构建，非 root 运行，uv 安装依赖。

## 7. 阶段交付计划

| 阶段 | 交付物 |
|---|---|
| 1（本阶段） | 本文档、ADR ×7、全部 domain 接口定义、异常体系、pyproject |
| 2 | core（配置/日志/安全）、infrastructure（PG/Redis/ARQ/迁移）、llm 适配层 + 重试熔断 |
| 3 | agents（ReAct / Planner-Executor 图、checkpoint、审批中断）、tools（注册中心/权限/内置工具） |
| 4 | memory（Redis/pgvector/压缩）、rag（解析/分块/混合检索/重排序） |
| 5 | api（路由/SSE/中间件/全局异常）、测试（≥80% 覆盖、LLM 全 Mock）、Docker Compose + K8s |
