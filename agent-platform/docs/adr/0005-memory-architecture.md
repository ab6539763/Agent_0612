# ADR-0005: 双层记忆 + 渐进式摘要压缩

- 状态：已接受
- 日期：2026-06-12

## 背景

Agent 需要：① 会话内低延迟的上下文窗口；② 跨会话的用户/事实记忆；③ 长对话时
控制 prompt token 成本。三者读写模式与一致性要求不同，不应揉进一个存储。

## 决策

1. **短期记忆 = Redis**（`ShortTermMemory` 接口）：
   - 每会话一个 List + 元数据 Hash，消息按序追加；读取按 token 预算从尾部取窗口。
   - TTL 跟随会话空闲时间；写入 O(1)，读取 O(窗口)，满足流式场景延迟要求。
   - 同时作为 LangGraph 之外的"对话历史真相源"（checkpoint 是执行态，不是
     对话历史的查询接口）。
2. **长期记忆 = pgvector**（`LongTermMemory` 接口）：
   - 条目 = 文本 + embedding + 类型（episodic/semantic/preference）+
     `tenant_id/user_id` 隔离 + 重要性评分与时间衰减。
   - 检索 = 向量相似度 × 衰减权重，Top-K 注入 Agent 上下文。
   - 选 pgvector 而非 Milvus：记忆量级（每用户千条级）远未到需要独立向量集群，
     复用 PG 省一套运维与备份体系；`LongTermMemory` 是接口，量级上来可平移 Milvus。
3. **压缩 = 独立 `MemoryCompressor` 接口，后台异步执行**：
   - 触发：会话消息 token 超过高水位（如预算的 80%）时投递 ARQ 任务。
   - 策略：渐进式摘要——`新摘要 = LLM(旧摘要 + 被挤出窗口的消息)`；摘要作为
     固定头部消息存回 Redis，被压缩的原始消息按规则蒸馏为长期记忆条目。
   - 在后台执行的原因：压缩是 LLM 调用（百 ms~s 级），不能阻塞对话主链路；
     压缩失败仅导致窗口暂时偏长，可安全重试。

## 备选方案

- **只用 LangGraph checkpoint 当记忆**：checkpoint 面向执行恢复，无 token 预算
  语义、无跨会话检索，否决。
- **全量消息进 PG、无 Redis**：读路径延迟与 PG 压力在高并发流式场景不可接受，否决。

## 后果

- 记忆读写与对话主链路解耦，压缩对用户透明。
- 需处理 Redis 与 PG 的最终一致（压缩任务幂等 + 以 Redis 版本号做乐观并发控制）。
