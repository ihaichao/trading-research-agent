"""主图：gather → plan →(并行)→ research × N → assemble。

1. **并行。** N 个 researcher 各自调一次模型，同时跑。总耗时约等于最慢那个，
   而不是全部之和。这也是把一次大调用拆成多次小调用的主要收益。
2. **可中断续跑。** 接了 checkpointer，任何一步之后都能停下、审查、再继续。
   长任务、人工确认、崩溃恢复都靠它。
3. **汇总不再需要模型。** researcher 产出的已经是校验过的"句子 + 出处"，
   assemble 只做分组和装配——省一次调用，也少一个出错的地方。
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from tra.agent.llm import LLMCallable
from tra.graph.nodes import gather, make_plan_node, make_research_node
from tra.graph.state import Finding, ResearchState, SubQuestion
from tra.report.schema import (
    Claim,
    MetricSeries,
    Report,
    ReportMeta,
    Section,
    SectionKind,
    Source,
)

log = logging.getLogger(__name__)

SECTION_TITLES = {
    SectionKind.SUMMARY: "Summary",
    SectionKind.BUSINESS: "Business",
    SectionKind.FINANCIALS: "Financial performance",
    SectionKind.VALUATION: "Valuation",
    SectionKind.CATALYSTS: "Recent catalysts",
    SectionKind.BULL_BEAR: "Bull and bear case",
    SectionKind.RISKS: "Risks",
}
SECTION_ORDER = list(SECTION_TITLES)

MAX_CLAIMS_PER_SECTION = 5
"""每段判断上限。三个 researcher 都碰到同一个异常时，这一段会堆到八条。"""

DUPLICATE_THRESHOLD = 0.6
"""两条判断的词集重合度超过这个值就算重复。"""


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.%]+", text.lower()) if len(w) > 2}


def _is_duplicate(text: str, kept: list[str]) -> bool:
    """判断这条发现是否和已保留的某条重复。

    并行 researcher 之间必然重叠——同一个财务异常会被三个人各讲一遍。
    告诉它们"别人在管什么"能减少但消不掉，因为 planner 拆出来的问题本身就交叉。

    所以在装配这一步兜底：**用词集重合度做去重，确定性，不花一次模型调用。**
    """
    words = _tokens(text)
    if not words:
        return False
    for other in kept:
        other_words = _tokens(other)
        if not other_words:
            continue
        overlap = len(words & other_words) / min(len(words), len(other_words))
        if overlap >= DUPLICATE_THRESHOLD:
            return True
    return False


def _finding_sort_key(finding: Finding) -> tuple[int, str, str]:
    """给发现一个稳定顺序。

    并行节点的返回顺序取决于谁先跑完，**同样的输入会产出顺序不同的报告**。
    评测集要比较两次运行的差异，就必须先消掉这个随机性。
    """
    section_index = SECTION_ORDER.index(finding.section) if finding.section in SECTION_ORDER else 99
    return (section_index, finding.dimension, finding.text)


def fan_out(state: ResearchState) -> list[Send]:
    """把每个子问题派给一个 researcher。

    返回一组 `Send` 就是 LangGraph 的扇出：它们并行执行，各自的返回值
    通过 state 上的 reducer 合并。**没有 reducer 这里会直接报并发更新错误。**
    """
    plan = [SubQuestion.model_validate(item) for item in state.get("plan") or []]
    return [
        Send(
            "research",
            {
                "ticker": state["ticker"],
                "company_name": state.get("company_name", state["ticker"]),
                "sub_question": sub.model_dump(mode="json"),
                "siblings": [other.question for other in plan if other.key != sub.key],
                "evidence": state["evidence"],
                "allowed_source_ids": state["allowed_source_ids"],
            },
        )
        for sub in plan
    ]


def _series(state: ResearchState) -> list[MetricSeries]:
    return [MetricSeries.model_validate(item) for item in state.get("series") or []]


def assemble(state: ResearchState) -> dict:
    """把并行回来的发现装配成段落。确定性，不碰模型。"""
    findings = state.get("findings") or []
    log.info("[assemble] 收到 %d 条发现", len(findings))
    return {"problems": []} if findings else {"problems": ["no findings produced"]}


def build_report(state: ResearchState, *, started: datetime | None = None) -> Report | None:
    """从终态构造 Report。图跑完之后调用。"""
    findings = [Finding.model_validate(item) for item in state.get("findings") or []]
    if not findings:
        return None

    started = started or datetime.now(UTC)
    by_section: dict[SectionKind, list[Claim]] = {}
    kept_text: dict[SectionKind, list[str]] = {}
    dropped = 0

    for finding in sorted(findings, key=_finding_sort_key):
        seen = kept_text.setdefault(finding.section, [])
        if len(seen) >= MAX_CLAIMS_PER_SECTION or _is_duplicate(finding.text, seen):
            dropped += 1
            continue
        seen.append(finding.text)
        by_section.setdefault(finding.section, []).append(
            Claim(
                text=finding.text,
                source_ids=finding.source_ids,
                confidence=finding.confidence,
            )
        )

    if dropped:
        log.info("[assemble] 去重与封顶丢弃 %d 条重复/超量的判断", dropped)

    sections: list[Section] = []
    for kind in SECTION_ORDER:
        claims = by_section.get(kind)
        if not claims:
            continue
        section = Section(kind=kind, title=SECTION_TITLES[kind], claims=claims)
        if kind == SectionKind.FINANCIALS:
            section.metrics = _series(state)
        sections.append(section)

    # 指标表没地方挂时，补一个财务段落，别把数据丢了
    if state.get("series") and not any(s.kind == SectionKind.FINANCIALS for s in sections):
        sections.append(
            Section(
                kind=SectionKind.FINANCIALS,
                title=SECTION_TITLES[SectionKind.FINANCIALS],
                metrics=_series(state),
            )
        )

    return Report(
        ticker=state["ticker"],
        company_name=state.get("company_name") or state["ticker"],
        question=state["question"],
        generated_at=started,
        data_as_of=started,
        sections=sections,
        sources=[Source.model_validate(item) for item in state.get("sources") or []],
        meta=ReportMeta(
            duration_seconds=(datetime.now(UTC) - started).total_seconds(),
            research_iterations=len(state.get("plan") or []),
        ),
    )


def build_graph(llm: LLMCallable) -> StateGraph:
    """搭图。编译留给调用方，因为 checkpointer 需要在上下文管理器里创建。"""
    graph: StateGraph = StateGraph(ResearchState)
    graph.add_node("gather", gather)
    graph.add_node("plan", make_plan_node(llm))
    graph.add_node("research", make_research_node(llm))
    graph.add_node("assemble", assemble)

    graph.add_edge(START, "gather")
    graph.add_edge("gather", "plan")
    graph.add_conditional_edges("plan", fan_out, ["research"])
    graph.add_edge("research", "assemble")
    graph.add_edge("assemble", END)
    return graph


def compile_graph(llm: LLMCallable, *, checkpointer: Any = None, pause_after_plan: bool = False):  # type: ignore[no-untyped-def]
    """编译成可执行的图。

    pause_after_plan=True 时会在规划之后停下，等人确认大纲再继续——
    这就是 human-in-the-loop。它依赖 checkpointer：没有持久化的状态，
    "停下来"就等于"丢掉进度"。
    """
    if pause_after_plan and checkpointer is None:
        raise ValueError("要在规划后暂停，必须提供 checkpointer——否则进度无处保存")
    return build_graph(llm).compile(
        checkpointer=checkpointer,
        interrupt_after=["plan"] if pause_after_plan else None,
    )
