"""工具的统一外壳。

1. **失败不抛异常，失败是一种返回值。**
   工具报错时，需要看到错误的是**模型**，不是调用栈。异常会中断循环；
   `ToolResult(ok=False)` 会变成一条 Observation 回灌给模型，让它自己纠正。

2. **数据和出处一起返回。**

3. **错误信息必须是可执行的。**
   `error` 说发生了什么，`hint` 说下一步能干什么。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field

from tra.report.schema import Source


class ToolResult[T](BaseModel):
    """任何工具的返回值。

    工具的 `data` 应该直接是**报告契约里的类型**（`MetricSeries` / `Source` ...）
    """

    ok: bool
    data: T | None = None
    error: str | None = None
    hint: str | None = Field(
        default=None,
        description="失败时给模型的下一步建议。没有它，模型只能重试同样的错误。",
    )
    sources: list[Source] = Field(default_factory=list)

    @classmethod
    def success(cls, data: T, sources: Sequence[Source] = ()) -> ToolResult[T]:
        return cls(ok=True, data=data, sources=list(sources))

    @classmethod
    def failure(cls, error: str, *, hint: str | None = None) -> ToolResult[T]:
        return cls(ok=False, error=error, hint=hint)

    @classmethod
    def from_exception(cls, exc: BaseException, *, hint: str | None = None) -> ToolResult[T]:
        """把异常转成失败结果。工具内部 `except Exception as e` 之后用它。

        带上异常类型名，模型能区分"网络超时"和"参数写错了"——前者该重试，
        后者该换参数。
        """
        return cls.failure(f"{type(exc).__name__}: {exc}", hint=hint)

    def for_model(self) -> str:
        """转成喂给 LLM 的文本。
        """
        if not self.ok:
            lines = [f"ERROR: {self.error or 'unknown error'}"]
            if self.hint:
                lines.append(f"HINT: {self.hint}")
            return "\n".join(lines)

        lines = [_render(self.data)]
        if self.sources:
            lines.append("SOURCES:")
            lines.extend(
                f"  [{s.id}] {s.title}" + (f" — {s.locator}" if s.locator else "")
                for s in self.sources
            )
        return "\n".join(line for line in lines if line)


def _render(data: Any) -> str:
    """把工具数据渲染成模型能读的文本。
    """
    if data is None:
        return ""
    if isinstance(data, str):
        return data
    if isinstance(data, BaseModel):
        return data.model_dump_json()
    if isinstance(data, list | tuple) and data and isinstance(data[0], BaseModel):
        items = ", ".join(item.model_dump_json() for item in data)
        return f"[{items}]"
    return str(data)
