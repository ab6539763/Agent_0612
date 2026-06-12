# ADR-0002: 使用 LangGraph StateGraph 做 Agent 编排

- 状态：已接受
- 日期：2026-06-12

## 背景

需要支持 ReAct 循环、Planner-Executor、多 Agent 协作三种模式，且必须支持
断点恢复（checkpoint）与 Human-in-the-loop 中断。LangChain 的 `AgentExecutor`
已废弃，不在考虑范围。

## 决策

1. 所有编排基于 LangGraph `StateGraph`：
   - **ReAct**：`reason → act → observe` 条件环，护栏（最大迭代/Token 预算）作为
     条件边而非节点内 if，便于在事件流中暴露终止原因。
   - **Planner-Executor**：`plan → route → executor → review → (replan|respond)`；
     executor 为子图，多个专职 Agent（研究/编码/写作等）注册为可路由子图，
     通过共享状态通道交换结果，实现多 Agent 协作。
2. **状态**：`AgentState` 用 Pydantic 模型声明，消息通道使用 reducer 追加合并；
   状态必须可 JSON 序列化以支持 checkpoint。
3. **Checkpointer**：`langgraph-checkpoint-postgres`（AsyncPostgresSaver），与业务
   库同实例、独立 schema；`thread_id = run_id`。
4. **Human-in-the-loop**：高危工具节点调用 LangGraph `interrupt()`，运行进入
   `waiting_approval`；审批 API 写入决策后以 `Command(resume=...)` 续跑。
5. **平台边界**：LangGraph 类型不泄漏出 `src/agents`，对外只暴露
   `AgentRuntime` Protocol 与 `AgentEvent` 流，保证上层（api/队列）不感知框架。

## 备选方案

- **自研状态机**：可控性最高，但 checkpoint、interrupt、流式事件都要重造，否决。
- **AutoGen / CrewAI**：多 Agent 会话抽象好，但 checkpoint/中断恢复能力与
  Postgres 持久化生态弱于 LangGraph，且与"图"心智模型不符，否决。

## 后果

- 断点恢复与审批中断由框架保证，平台只维护状态 Schema 的演进兼容。
- 需要把 LangGraph 的事件（`astream_events`）翻译为平台 `AgentEvent`，
  该翻译层是阶段 3 的关键交付物。
