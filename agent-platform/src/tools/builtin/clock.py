"""当前时间工具。"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from src.core.exceptions import ToolExecutionError
from src.core.types import ToolPermission
from src.tools.base import BaseTool, ToolContext


class CurrentTimeArgs(BaseModel):
    """时间查询参数。"""

    timezone: str = Field(
        default="UTC",
        description="IANA 时区名，如 'Asia/Shanghai'、'America/New_York'。",
    )


class CurrentTimeTool(BaseTool):
    """查询指定时区的当前时间（模型没有可靠的实时时钟）。"""

    name: ClassVar[str] = "current_time"
    description: ClassVar[str] = "获取指定时区的当前日期与时间（ISO 8601 格式）。"
    args_schema: ClassVar[type[BaseModel]] = CurrentTimeArgs
    required_permission: ClassVar[ToolPermission] = ToolPermission.READ

    async def run(self, args: BaseModel, context: ToolContext) -> str:
        """返回当前时间。

        Raises:
            ToolExecutionError: 时区名不合法。
        """
        assert isinstance(args, CurrentTimeArgs)
        try:
            zone = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolExecutionError(
                f"unknown timezone: {args.timezone}", cause=exc
            ) from exc
        now = datetime.now(tz=UTC).astimezone(zone)
        return f"{now.isoformat()} ({args.timezone})"
