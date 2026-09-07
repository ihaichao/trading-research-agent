"""端到端流水线：ticker -> 一份带引用的 Report。

分工是这一层的全部意义：

    取数    程序做（M1 的工具）
    计算    程序做（calc.py，一步都不给模型）
    判断    模型做（哪些事实值得说、怎么组织）
    装配    程序做（数字直接进 Report，模型碰不到）
    校验    schema 做（悬空引用直接构造失败）

模型在这条链路里唯一的产出是**句子和它的出处标注**。数字从 SEC 到
report.json 全程没经过它的手。
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from tra.agent.evidence import EvidencePack
from tra.agent.llm import LLMCallable
from tra.agent.synthesize import draft_sections
from tra.report.schema import Report, ReportMeta, Section, SectionKind
from tra.tools import calc, get_quote
from tra.tools.base import ToolResult
from tra.tools.fundamentals import COMPANY_NAMES, CONCEPTS, get_financials

log = logging.getLogger(__name__)

DEFAULT_QUESTION = "How has the business been performing recently, and what are the main risks?"


def collect_evidence(ticker: str, *, periods: int = 12) -> tuple[EvidencePack, ToolResult]:
    """跑工具，攒出一份证据包。"""
    log.info("正在取 %s 的 SEC 财务数据（首次会下载 companyfacts，可能要几十秒）…", ticker)
    started = time.monotonic()
    financials = get_financials(ticker, metrics=tuple(CONCEPTS), periods=periods)
    log.info("SEC 取数完成，用时 %.1fs", time.monotonic() - started)

    symbol = ticker.upper()
    # 取过数之后工具层会把 SEC 登记的公司名填进来；没取到就退回代码本身
    pack = EvidencePack(ticker=symbol, company_name=COMPANY_NAMES.get(symbol, symbol))
    if financials.ok and financials.data:
        reported = financials.data
        pack.series = [*reported, *calc.derive_all({s.key: s for s in reported})]
        pack.sources = list(financials.sources)
        if financials.hint:
            pack.notes.append(financials.hint)
    else:
        pack.notes.append(f"fundamentals unavailable: {financials.error}")

    pack.company_name = COMPANY_NAMES.get(symbol, symbol)

    log.info("正在取行情…")
    quote = get_quote(ticker)
    if quote.ok and quote.data:
        pack.sources.extend(quote.sources)
        pack.notes.append(
            f"latest price {quote.data.price:,.2f} {quote.data.currency} "
            f"(delayed, cite {quote.sources[0].id})"
        )
    else:
        pack.notes.append("market quote unavailable; do not state a price")

    return pack, financials


def attach_metrics(sections: list[Section], pack: EvidencePack) -> list[Section]:
    """把指标序列直接挂到 financials 段落上。"""
    if not pack.series:
        return sections

    for section in sections:
        if section.kind == SectionKind.FINANCIALS:
            section.metrics = list(pack.series)
            return sections

    sections.append(
        Section(
            kind=SectionKind.FINANCIALS,
            title="Financial performance",
            metrics=list(pack.series),
        )
    )
    return sections


def research(
    ticker: str,
    *,
    question: str = DEFAULT_QUESTION,
    llm: LLMCallable,
    periods: int = 12,
) -> ToolResult[Report]:
    started = datetime.now(UTC)
    symbol = ticker.strip().upper()

    pack, financials = collect_evidence(symbol, periods=periods)
    if not pack.sources:
        return ToolResult.failure(
            f"no evidence could be gathered for {symbol!r}: {financials.error}",
            hint="check the ticker, or that SEC/market data sources are reachable.",
        )

    log.info("证据包就绪：%d 条指标、%d 个出处", len(pack.series), len(pack.sources))
    drafting_started = time.monotonic()
    sections, problems = draft_sections(pack, question, llm)
    log.info("起草完成，用时 %.1fs", time.monotonic() - drafting_started)
    if not sections:
        return ToolResult.failure(
            f"the model could not produce a citable draft for {symbol!r}: {problems}",
            hint="every claim must cite a source id from the evidence pack.",
        )

    sections = attach_metrics(sections, pack)

    report = Report(
        ticker=symbol,
        company_name=pack.company_name,
        question=question,
        generated_at=started,
        data_as_of=started,
        sections=sections,
        sources=pack.sources,
        meta=ReportMeta(
            duration_seconds=(datetime.now(UTC) - started).total_seconds(),
            research_iterations=len(problems) + 1,
        ),
    )

    result: ToolResult[Report] = ToolResult.success(report, pack.sources)
    if problems:
        result.hint = f"draft was rejected {len(problems)} time(s): {problems}"
    return result
