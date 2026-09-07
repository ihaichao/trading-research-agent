"""图的状态定义。

这是整个 M3 里最需要想清楚的东西——**节点之间只通过 state 通信**，
state 的形状决定了哪些节点能并行、哪些必须串行。

1. **会被并行写入的字段必须配 reducer。** `findings` 由 N 个 researcher
   同时写，不加 `Annotated[..., add]` 会直接报并发更新错误。这是 fan-out
   最常踩的坑，也是"图"和"一条直线"的第一个真实区别。

2. **state 里只放纯数据（dict / str / 数字），不放 pydantic 对象。**
   checkpointer 要把 state 序列化落盘，塞进自定义类型会得到
   "Deserializing unregistered type ... will be blocked in a future version" 的告警——
   持久化的东西必须是**跨版本、跨进程都能读回来**的。
   做法是：**边界处校验成模型，进 state 前 dump 成 dict**。类型安全一点没丢，
   丢的只是"把对象直接塞进去"的方便。

3. **原始材料不进 state。** state 里只放压缩后的结论和引用 ID。
   M2 那条直线之所以能把 12 个季度全塞给模型，是因为数据小；
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from tra.report.schema import SectionKind


class SubQuestion(BaseModel):
    """一个研究子问题。plan 节点产出，researcher 节点消费。"""

    key: str = Field(description="稳定标识，用于日志和去重")
    section: SectionKind = Field(description="这个子问题的结论归到报告哪一段")
    question: str = Field(min_length=1)


class Finding(BaseModel):
    """一条经过校验的发现。researcher 的产出单位。

    它已经是"可以直接进报告的句子 + 出处"，所以汇总那一步**不需要再调模型**——
    这是把 M2 的一次大调用拆成 N 次小调用之后白拿的好处。
    """

    section: SectionKind
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1, max_length=3)
    confidence: Literal["high", "medium", "low"] = "medium"
    dimension: str = Field(default="", description="来自哪个子问题，便于追溯")


class ResearchState(TypedDict, total=False):
    """主图状态。"""

    ticker: str
    question: str
    company_name: str

    # 证据（由 gather 节点一次性备齐，所有 researcher 共享）
    evidence: str
    allowed_source_ids: list[str]

    # 下面这些都是 model_dump(mode="json") 之后的纯 dict，
    # 用的时候再 model_validate 回来。见文件头第 2 条。
    series: list[dict]
    sources: list[dict]
    plan: list[dict]

    # ↓ 并行写入，必须有 reducer
    # 告诉 LangGraph：当有节点返回 findings 时，不要覆盖，而是用 add（列表拼接）合并！
    findings: Annotated[list[dict], add]
    problems: Annotated[list[str], add]


class ResearcherInput(TypedDict):
    """Send 给单个 researcher 的载荷。

    刻意不是完整的 ResearchState：researcher 只该看到它这一路需要的东西。
    收窄输入既省 token，也让"这个节点依赖什么"一目了然。
    """

    ticker: str
    company_name: str
    sub_question: dict
    siblings: list[str]
    """其他 researcher 负责的问题。告诉它别人在管什么，减少重复劳动。"""
    evidence: str
    allowed_source_ids: list[str]
