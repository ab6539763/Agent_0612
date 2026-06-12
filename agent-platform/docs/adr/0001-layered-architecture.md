# ADR-0001: 分层架构与依赖倒置

- 状态：已接受
- 日期：2026-06-12

## 背景

平台包含编排、工具、记忆、检索、模型接入等多个会独立演进的子系统，且基础设施
（向量库、队列、模型厂商）存在替换可能。需要一个能约束依赖方向、便于单元测试
（Mock 边界清晰）的结构。

## 决策

采用分层结构：`api → (agents|tools|memory|rag) → llm → infrastructure → core`。

1. **接口归属各模块的 `base.py`**：`agents/tools/memory/rag/llm` 在 `base.py`
   中定义 Protocol/ABC 与 Pydantic 领域模型，`base.py` 只允许依赖其他模块的
   `base.py` 与 core（零基础设施依赖）。模块的具体实现（如
   `memory/redis_short_term.py`）可使用 `infrastructure` 提供的客户端，
   但跨模块只消费接口。
2. **infrastructure 提供能力而非业务**：数据库引擎/会话、Redis 客户端、队列、
   事件流、通用仓储；不包含业务逻辑，不反向依赖业务模块。
3. **组合根**：所有实现的装配只发生在 `src/api/deps.py`（FastAPI 依赖注入）与
   worker 入口，业务代码不出现 `RedisShortTermMemory()` 这类跨模块直接实例化。
4. **静态校验**：import-linter contracts + ruff banned-api 在 CI 阻断违规 import。

## 备选方案

- **不分层（扁平模块）**：初期快，但 LLM/向量库替换与测试 Mock 成本随规模上升，否决。
- **完整六边形架构（ports/adapters 独立目录）**：表达力更强但目录噪音大，
  与任务给定的项目结构不符，否决；本方案保留了其核心（依赖倒置），简化了形式。

## 后果

- 单元测试只需实现 Protocol 的 Fake/Mock，不需要起真实依赖。
- 新增模型厂商/向量库 = 新增一个 infrastructure 实现 + 组合根一行装配。
- 代价：接口与模型定义有少量前期成本（即本阶段产出）。
