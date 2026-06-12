# ADR-0003: 自研轻量 LLM 适配层，litellm 作为其中一个 Provider

- 状态：已接受
- 日期：2026-06-12

## 背景

需要统一接入 OpenAI / Anthropic / 本地 vLLM，并实现重试（指数退避）与熔断。
候选：直接全面依赖 litellm，或自研适配接口。

## 决策

**自研薄接口 + litellm 兜底**：

1. 平台定义 `LLMProvider` Protocol（`complete` / `stream`）与自有消息模型
   （`ChatMessage` / `CompletionResult` / `CompletionChunk` / `ToolCallRequest`），
   所有上层代码只面向该接口。
2. 一级厂商（OpenAI、Anthropic）用官方 async SDK 实现 Provider —— 类型完整、
   流式工具调用语义可控、新特性跟进快。
3. `LiteLLMProvider` 作为通用实现覆盖 vLLM（OpenAI-compatible）及其他长尾厂商，
   避免为每个厂商写适配器。
4. 弹性策略不写进 Provider，而是装饰器组合（均实现同一 Protocol）：

   ```
   ProviderRouter → CircuitBreakerProvider → RetryingProvider → 具体 Provider
   ```

   - 重试：仅对 `retryable=True` 的异常（超时、429、5xx），指数退避 + 全抖动，
     默认 3 次，预算上限受请求 deadline 约束。
   - 熔断：滑动窗口错误率阈值触发开启，半开态放行探测请求；按
     `provider+model` 维度独立熔断；状态导出 Prometheus 指标。
5. 错误归一化：各厂商异常在 Provider 内翻译为平台 `LLMError` 子类，上层不出现
   `openai.RateLimitError` 之类的厂商类型。

## 备选方案

- **纯 litellm**：接入快，但其异常归一化与流式 chunk 结构历史上多次变更，作为
  全平台唯一边界风险过高；保留为其中一个实现可两全。
- **每厂商裸 SDK 无统一层**：上层充斥厂商分支，否决。

## 后果

- 上层（agents/rag/memory）完全不感知厂商；新增厂商成本 = 一个 Provider 类。
- 需自行维护 OpenAI/Anthropic 两个适配器的流式工具调用解析（阶段 2 交付，
  配套契约测试防止 SDK 升级破坏）。
