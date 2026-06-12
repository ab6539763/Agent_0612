# ADR-0007: 统一 AgentEvent 事件协议与 SSE 传输

- 状态：已接受
- 日期：2026-06-12

## 背景

要求 Agent 输出经 SSE 流式返回且包含中间推理步骤。如果直接把 LangGraph 的
`astream_events` 原始事件透传给客户端，客户端将耦合框架内部结构（节点名、
命名空间），且无法做脱敏与权限裁剪。

## 决策

1. **平台自有事件模型**：`AgentEvent` 为 Pydantic 判别联合（`type` 字段判别），
   事件类型见架构文档 3.6 节。LangGraph 原始事件在 `src/agents` 内翻译为
   `AgentEvent`，框架细节不出模块边界。
2. **传输 = SSE**（而非 WebSocket）：单向流场景 SSE 足够，过代理/网关友好、
   自带断线重连语义；需要双向交互的审批走独立 REST 端点，不复用流通道。
3. **断线续传**：每个事件带单调递增 `seq`，SSE `id:` 字段输出；事件同时写入
   Redis Stream（按 run 维度，TTL 1h）。客户端携带 `Last-Event-ID` 重连时从
   Stream 补发。这也使"任务在 worker 执行、事件在 API 节点下发"的跨进程
   流式成为可能（worker 写 Stream，API 读）。
4. **保活与终止**：15s 无事件发 `heartbeat`；任何终态（`run_finished` /
   `run_failed`）后服务端关闭流。错误不走 HTTP 状态码（头已发出），统一以
   `run_failed` 事件承载 Problem Details。
5. **脱敏与裁剪**：`tool_call.arguments` / `tool_result.content` 经 core 脱敏
   processor 处理后才能进入事件；`reasoning_delta` 可按租户配置关闭。

## 备选方案

- **WebSocket**：双向能力本场景用不上，且网关/LB 配置成本更高，否决。
- **轮询任务状态**：失去 token 级流式体验，仅保留为降级路径（任务结果可经
  REST 查询）。

## 后果

- 客户端只面向稳定的事件 Schema，平台可自由更换编排框架。
- Redis Stream 引入少量写放大，换来跨进程流式与断线续传，可接受。
