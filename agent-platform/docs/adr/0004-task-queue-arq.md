# ADR-0004: 任务队列选用 ARQ（而非 Celery）

- 状态：已接受
- 日期：2026-06-12

## 背景

长任务（深度研究类 Agent 运行、批量文档摄取、记忆压缩）需要异步执行，要求：
asyncio 原生、支持任务重投与延迟任务、可观测、与 Redis 复用。平台全链路 async，
任务体内会大量 await LLM 与数据库。

## 决策

选用 **ARQ**（Redis-backed，asyncio 原生）。

1. **asyncio 一致性**：ARQ worker 即事件循环，任务函数就是协程，直接复用平台的
   async 数据库会话与 LLM Provider。Celery 任务体是同步模型，跑 async 代码需要
   每任务管理事件循环或经 gevent/eventlet，复杂且易踩 SQLAlchemy async 兼容坑。
2. **可靠性边界明确**：ARQ 提供 at-least-once（任务超时重投）、`job_id` 幂等、
   延迟任务、定时任务，覆盖本平台全部需求。
3. **业务状态机不依赖队列**：任务的 `queued/running/waiting_approval/...` 状态
   持久化在 PostgreSQL `task_runs` 表，队列只负责"触发执行"。断点恢复 =
   重投任务 + LangGraph checkpoint 续跑，与队列实现解耦——未来若需更换为
   Celery/Temporal，业务层不变。
4. **审批暂停不占用 worker**：进入 `waiting_approval` 时任务正常返回（checkpoint
   已落盘），审批通过后由 API 重新入队恢复任务，避免长期占用并发槽位。

## 备选方案

- **Celery**：生态最大、监控工具成熟（Flower），但 async 支持是二等公民（见上），
  且本平台不需要其多 broker/复杂路由能力，否决。
- **Temporal**：工作流语义最强（durable execution），但引入独立服务集群，
  运维成本对当前规模过重；状态机 + checkpoint 方案已覆盖需求。记录为未来选项。

## 后果

- 运维面只有 Redis，无新增组件。
- ARQ 社区较 Celery 小；通过把队列收敛在 `TaskQueue` Protocol 之后控制替换成本。
- 需要自建队列深度/任务时延的 Prometheus 指标（ARQ 无内置 exporter，阶段 2 实现）。
