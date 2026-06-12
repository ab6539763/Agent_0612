# ADR-0006: RAG 混合检索（pgvector + PG 全文检索，RRF 融合）+ 重排序

- 状态：已接受
- 日期：2026-06-12

## 背景

纯向量检索对专有名词、编号、精确短语召回差；纯关键词检索缺语义泛化。需要混合
检索 + 重排序，并决定向量库与关键词引擎选型。

## 决策

1. **存储统一在 PostgreSQL**：
   - 向量：`chunks.embedding vector(dim)`，HNSW 索引，余弦距离。
   - 关键词：`chunks.tsv tsvector` + GIN 索引，`ts_rank_cd` 打分作为 BM25 近似；
     中文经分词器（zhparser，部署在 PG 镜像中）。
   - 理由：文档/分块/权限/租户隔离本来就在 PG，单库事务保证索引与数据一致；
     当前规模（百万分块级）pgvector HNSW 足够。`Retriever` 是接口，超过该量级
     可平移 Milvus + Elasticsearch 而不动上层。
2. **融合 = RRF（Reciprocal Rank Fusion）**：`score = Σ 1/(k + rank_i)`，k=60。
   只依赖排名不依赖分数，无需对两路异构分数做归一化标定，对参数不敏感、实现简单。
3. **重排序 = 独立 `Reranker` 接口**：两路融合取 Top-50 后重排取 Top-K。
   内置实现：cross-encoder（bge-reranker，本地推理或独立服务）与
   LLM-as-reranker（低 QPS 场景）。重排可按请求关闭以换延迟。
4. **管道每阶段独立 Protocol**（Parser/Chunker/Embedder/Indexer/Retriever/
   Reranker），摄取走 ARQ 异步任务（解析与向量化是重操作），查询路径全同步 async。
5. **RAG 作为工具暴露**：`knowledge_search` 注册进 ToolRegistry，由 Agent 在
   ReAct 循环中按需调用（agentic RAG），而非每轮强制前置检索。

## 备选方案

- **Milvus + Elasticsearch**：能力上限更高，但引入两套有状态集群，当前规模
  收益不抵运维成本；接口已为迁移留位。
- **加权线性融合**：需要分数归一化与权重调参，跨语料稳定性差于 RRF，否决。

## 后果

- 单一 PG 承载业务 + 向量 + 全文，备份/迁移/权限模型统一。
- ts_rank 并非真 BM25（无文档长度归一化的 k1/b 参数），对召回排名影响有限
  （有重排序兜底）；如需精确 BM25 可引入 pg_search/VectorChord-bm25 扩展，
  记录为可选优化。
