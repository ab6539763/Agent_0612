"""RAG 管道接口定义（ADR-0006）。

摄取链：``DocumentParser → Chunker → (EmbeddingProvider) → Indexer``
查询链：``HybridRetriever（向量 + 全文, RRF 融合） → Reranker``

每个阶段独立 Protocol，可单独替换；实现位于阶段 4。
RAG 对 Agent 以 ``knowledge_search`` 工具形式暴露（agentic RAG）。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 领域模型
# ---------------------------------------------------------------------------


class DocumentFormat(StrEnum):
    """支持的源文档格式。"""

    PDF = "pdf"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"


class ParsedDocument(BaseModel):
    """解析后的规范化文档。"""

    model_config = ConfigDict(frozen=True)

    source_uri: str = Field(description="来源标识（文件名 / URL / 对象存储 key）。")
    format: DocumentFormat
    title: str | None = None
    text: str = Field(description="抽取出的全文（保留段落结构的纯文本/Markdown）。")
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """文档分块。"""

    model_config = ConfigDict(frozen=True)

    id: UUID
    document_id: UUID
    ordinal: int = Field(ge=0, description="块在文档内的序号。")
    text: str
    heading_path: tuple[str, ...] = Field(
        default=(),
        description="标题层级路径（如 ('部署', 'Kubernetes')），随块注入上下文。",
    )
    metadata: dict[str, str] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """检索命中的分块。"""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float = Field(description="当前阶段的相关性分（RRF 融合分或重排分）。")
    source: str = Field(
        description="召回来源标记：'vector' / 'fulltext' / 'fused' / 'reranked'。"
    )


# ---------------------------------------------------------------------------
# 摄取链接口
# ---------------------------------------------------------------------------


@runtime_checkable
class DocumentParser(Protocol):
    """文档解析接口。每种格式一个实现，由 ParserRegistry 按格式分发。"""

    @property
    def supported_formats(self) -> frozenset[DocumentFormat]:
        """本解析器支持的格式集合。"""
        ...

    async def parse(self, raw: bytes, *, source_uri: str) -> ParsedDocument:
        """解析原始字节为规范化文档。

        Args:
            raw: 原始文件内容。
            source_uri: 来源标识（写入文档元数据）。

        Returns:
            规范化文档。

        Raises:
            DocumentParseError: 内容损坏或格式不符。
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """分块接口。默认实现：递归字符分块，保留标题路径。"""

    def chunk(self, document: ParsedDocument, *, document_id: UUID) -> list[Chunk]:
        """把文档切分为块（纯 CPU 操作，无 IO，故为同步方法）。

        Args:
            document: 解析后的文档。
            document_id: 已持久化的文档 ID。

        Returns:
            按 ordinal 升序的分块列表。
        """
        ...


@runtime_checkable
class Indexer(Protocol):
    """索引写入接口：持久化分块、向量与全文索引（单事务）。"""

    async def index(self, *, tenant_id: UUID, chunks: list[Chunk]) -> int:
        """写入/更新分块索引。

        实现负责调用 EmbeddingProvider 批量向量化、写入 pgvector 列与
        tsvector 列；按 ``(document_id, ordinal)`` 幂等 upsert。

        Returns:
            实际写入的分块数。

        Raises:
            RetrievalError: 向量化或持久化失败。
        """
        ...

    async def remove_document(self, *, tenant_id: UUID, document_id: UUID) -> None:
        """删除文档的全部分块索引。"""
        ...


# ---------------------------------------------------------------------------
# 查询链接口
# ---------------------------------------------------------------------------


class RetrievalQuery(BaseModel):
    """一次检索请求。"""

    model_config = ConfigDict(frozen=True)

    tenant_id: UUID
    query: str
    top_k: int = Field(default=8, gt=0, le=50, description="最终返回条数。")
    candidate_k: int = Field(
        default=50, gt=0, le=200, description="融合后送入重排的候选条数。"
    )
    document_ids: tuple[UUID, ...] | None = Field(
        default=None, description="限定检索范围的文档集合；None 表示全库。"
    )
    rerank: bool = Field(default=True, description="是否启用重排序。")


@runtime_checkable
class Retriever(Protocol):
    """检索接口。混合实现：向量 + 全文两路并发召回，RRF 融合。"""

    async def retrieve(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """执行检索（不含重排）。

        Returns:
            按融合分降序、最多 ``candidate_k`` 条的候选分块。

        Raises:
            RetrievalError: 检索执行失败。
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """重排序接口（cross-encoder 或 LLM-as-reranker 实现）。"""

    async def rerank(
        self, query: str, candidates: list[RetrievedChunk], *, top_k: int
    ) -> list[RetrievedChunk]:
        """对候选分块重排。

        Args:
            query: 原始查询。
            candidates: 融合后的候选。
            top_k: 返回条数。

        Returns:
            按重排分降序的前 top_k 条。

        Raises:
            RetrievalError: 重排模型调用失败（调用方可降级为跳过重排）。
        """
        ...


@runtime_checkable
class RAGPipeline(Protocol):
    """对外的管道门面：组合摄取链与查询链。"""

    async def ingest(
        self,
        *,
        tenant_id: UUID,
        raw: bytes,
        source_uri: str,
        format: DocumentFormat,
    ) -> UUID:
        """摄取一篇文档（解析 → 分块 → 向量化 → 索引）。

        重操作，由 ARQ 任务调用；按 ``(tenant_id, source_uri)`` 幂等。

        Returns:
            文档 ID。

        Raises:
            DocumentParseError: 解析失败。
            RetrievalError: 索引失败。
        """
        ...

    async def search(self, query: RetrievalQuery) -> list[RetrievedChunk]:
        """端到端查询（检索 + 可选重排）。

        Returns:
            最终结果（``rerank=True`` 且重排失败时降级返回融合结果）。
        """
        ...
