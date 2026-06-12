"""HTTP 抓取工具（带 SSRF 基础防护）。"""

from __future__ import annotations

import ipaddress
from typing import ClassVar
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from src.core.exceptions import ToolExecutionError
from src.core.types import ToolPermission
from src.tools.base import BaseTool, ToolContext

_MAX_RESPONSE_BYTES = 512 * 1024
_BLOCKED_HOSTNAMES = frozenset({"localhost", "metadata.google.internal"})


def _validate_url(url: str) -> None:
    """拒绝明显的 SSRF 向量。

    覆盖：非 http(s) 协议、userinfo 混淆、localhost / 私有与保留 IP 字面量。
    DNS 解析到内网地址的变体需在网络层（egress 代理 / NetworkPolicy）兜底，
    见部署文档。

    Raises:
        ToolExecutionError: URL 不允许访问。
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ToolExecutionError(f"scheme not allowed: {parsed.scheme or '(empty)'}")
    if parsed.username or parsed.password:
        raise ToolExecutionError("userinfo in URL is not allowed")
    hostname = parsed.hostname or ""
    if not hostname or hostname.lower() in _BLOCKED_HOSTNAMES:
        raise ToolExecutionError(f"host not allowed: {hostname or '(empty)'}")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return  # 域名：字面量检查通过
    if (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
    ):
        raise ToolExecutionError(f"ip range not allowed: {hostname}")


class HttpFetchArgs(BaseModel):
    """抓取参数。"""

    url: str = Field(max_length=2048, description="要抓取的 http(s) URL。")


class HttpFetchTool(BaseTool):
    """抓取公网网页/接口的文本内容（GET）。"""

    name: ClassVar[str] = "http_fetch"
    description: ClassVar[str] = (
        "通过 HTTP GET 抓取公网 URL 的文本内容（最大 512KB）。"
        "适用于读取网页、API 与文档；不能访问内网地址。"
    )
    args_schema: ClassVar[type[BaseModel]] = HttpFetchArgs
    required_permission: ClassVar[ToolPermission] = ToolPermission.NETWORK

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """初始化工具。

        Args:
            client: 注入的 HTTP 客户端（测试 / 连接池复用）；缺省自建。
        """
        self._client = client or httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(10.0),
            limits=httpx.Limits(max_connections=10),
        )

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        """抓取 URL。

        Raises:
            ToolExecutionError: URL 被策略拒绝或请求失败。
        """
        assert isinstance(args, HttpFetchArgs)
        _validate_url(args.url)
        try:
            response = await self._client.get(args.url)
        except httpx.HTTPError as exc:
            raise ToolExecutionError(f"request failed: {exc}", cause=exc) from exc
        if response.status_code >= 400:
            raise ToolExecutionError(
                f"upstream returned HTTP {response.status_code}",
                details={"url": args.url},
            )
        body = response.text[:_MAX_RESPONSE_BYTES]
        content_type = response.headers.get("content-type", "unknown")
        return f"[{response.status_code}, {content_type}]\n{body}"
