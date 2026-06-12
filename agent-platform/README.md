# Agent Platform

企业级 AI Agent 平台：多 Agent 编排（LangGraph）、可插拔工具系统、双层记忆、
RAG 混合检索、异步长任务与 Human-in-the-loop 审批，SSE 全程流式输出。

## 文档

- [架构设计文档](docs/architecture.md)
- 架构决策记录（ADR）：[docs/adr/](docs/adr/)

## 当前状态

**阶段 3 / 5**：agents（LangGraph 编排）+ tools 系统已交付。

| 阶段 | 内容 | 状态 |
|---|---|---|
| 1 | 架构文档、ADR、domain 接口、异常体系 | 完成 |
| 2 | core（配置/日志/安全/可观测性）、infrastructure（PG/Redis/ARQ/迁移）、llm 适配层（重试+熔断） | 完成 |
| 3 | agents（ReAct / Planner-Executor 图、checkpoint、审批中断恢复）、tools（注册中心/权限/执行器/内置工具） | 完成 |
| 4 | memory、rag 管道 | 待开始 |
| 5 | api 层、SSE、测试、部署配置 | 待开始 |

## 接口导览

| 文件 | 契约 |
|---|---|
| `src/core/exceptions.py` | 全平台异常体系（code / retryable / Problem Details） |
| `src/core/types.py` | `Principal` / `ToolPermission` / `TokenUsage` |
| `src/llm/base.py` | `LLMProvider` / `EmbeddingProvider`、统一消息与流式 chunk 模型 |
| `src/tools/base.py` | `BaseTool` / `ToolRegistry` / `ToolExecutor`、权限模型 |
| `src/memory/base.py` | `ShortTermMemory` / `LongTermMemory` / `MemoryCompressor` |
| `src/rag/base.py` | 解析 / 分块 / 索引 / 混合检索 / 重排序 / `RAGPipeline` |
| `src/agents/base.py` | `AgentRuntime` / `AgentState` / `ApprovalGate`、运行状态机 |
| `src/agents/events.py` | `AgentEvent` 流式事件判别联合（SSE 协议） |
| `src/infrastructure/base.py` | `UnitOfWork` / `DistributedLock` / `EventStream` / `TaskQueue` |

## 开发环境

```bash
# Python 3.11+，推荐 uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# 配置
cp .env.example .env   # 按需填入 LLM 凭证等

# 静态检查与测试
ruff check src tests
mypy
lint-imports                 # 分层依赖契约
pytest                       # 单元测试（LLM 全 Mock，覆盖率门槛 80%）

# 数据库迁移（需要 PostgreSQL + pgvector）
alembic upgrade head
alembic upgrade head --sql   # 仅生成 SQL，不连库
```
