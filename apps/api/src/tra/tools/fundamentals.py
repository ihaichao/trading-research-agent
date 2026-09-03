"""基本面数据：把 SEC XBRL 变成报告契约里的 MetricSeries。

分三层，边界是刻意划的：

    _fetch_facts()   网络层。调 edgartools，薄，只做取数和字段归一化。
    build_series()   纯函数。去重、挑期间、算单位、生成 Source。可离线测。
    get_financials() 工具入口。组装 ToolResult，捕获一切异常。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal

from tra.config import get_settings
from tra.report.schema import MetricPoint, MetricSeries, Source, SourceKind, make_source_id
from tra.tools.base import ToolResult
from tra.tools.ratelimit import sec_limiter

# --------------------------------------------------------------------------
# 指标定义
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConceptSpec:
    """一个指标：对外的稳定 key，以及在 XBRL 里可能的叫法。

    候选名按常见度排序，取第一个有数据的。ASC 606（2018）之后大部分公司
    改用 RevenueFromContractWithCustomer*，老年份还留着 SalesRevenueNet。
    """

    key: str
    label: str
    unit: Literal["USD", "USD_millions", "percent", "ratio", "shares"]
    candidates: tuple[str, ...]


CONCEPTS: dict[str, ConceptSpec] = {
    "revenue": ConceptSpec(
        key="revenue",
        label="Total revenue",
        unit="USD_millions",
        candidates=(
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
            "SalesRevenueGoodsNet",
        ),
    ),
    "cost_of_revenue": ConceptSpec(
        key="cost_of_revenue",
        label="Cost of revenue",
        unit="USD_millions",
        candidates=("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    ),
    "gross_profit": ConceptSpec(
        key="gross_profit",
        label="Gross profit",
        unit="USD_millions",
        candidates=("GrossProfit",),
    ),
    "operating_income": ConceptSpec(
        key="operating_income",
        label="Operating income",
        unit="USD_millions",
        candidates=("OperatingIncomeLoss",),
    ),
    "net_income": ConceptSpec(
        key="net_income",
        label="Net income",
        unit="USD_millions",
        candidates=("NetIncomeLoss", "ProfitLoss"),
    ),
    "rnd_expense": ConceptSpec(
        key="rnd_expense",
        label="R&D expense",
        unit="USD_millions",
        candidates=("ResearchAndDevelopmentExpense",),
    ),
}

DEFAULT_METRICS = ("revenue", "gross_profit", "operating_income", "net_income")


@dataclass(frozen=True)
class RawFact:
    concept: str
    """带 taxonomy 前缀的全名，如 us-gaap:CostOfRevenue。"""
    value: float
    period_start: date | None
    period_end: date
    fiscal_year: int | None
    fiscal_period: str | None
    form: str | None
    accession: str | None
    filed: date | None


def concept_name(concept: str) -> str:
    """去掉 taxonomy 前缀：us-gaap:CostOfRevenue -> CostOfRevenue。

    edgartools 的索引以带前缀的全名为 key，我们的候选清单写的是裸名。
    所有比较都在归一化之后做。
    """
    return concept.split(":")[-1]


def period_label(fact: RawFact) -> str:
    """期间标签，如 FY2026Q2。

    优先用 XBRL 自带的财年/财季——**不能自己按自然季度推**：
    NVDA 的财年一月底结束，FY2026Q2 覆盖的是自然年的 5–7 月。
    """
    if fact.fiscal_year and fact.fiscal_period:
        fp = fact.fiscal_period.upper()
        return f"FY{fact.fiscal_year}{fp}" if fp != "FY" else f"FY{fact.fiscal_year}"
    return fact.period_end.isoformat()


def filing_url(cik: int | str, accession: str) -> str:
    """accession → 可点开的 SEC 原文索引页。

    目录名去掉横杠，文件名保留横杠。这个不对称是 EDGAR 的历史遗留：
        https://www.sec.gov/Archives/edgar/data/1045810/000104581025000123/0001045810-25-000123-index.html
    """
    cik_plain = str(int(cik))
    no_dash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_plain}/{no_dash}/{accession}-index.html"


def dedupe_latest_filed(facts: list[RawFact]) -> list[RawFact]:
    """同一期间保留 filed 最新的那条。

    为什么会重复：FY2026Q2 的营收会出现在当季 10-Q、一年后 10-Q 的对比列、
    以及 10-K 的对比列里，三到五条。
    为什么取最新：最新那份 filing 反映公司当下对该期间的最终认定，
    重述（restatement）已经体现在里面。做投资研究要的是"现在认为当时是多少"。
    """
    best: dict[date, RawFact] = {}
    for fact in facts:
        current = best.get(fact.period_end)
        if current is None:
            best[fact.period_end] = fact
            continue
        # filed 缺失时排在后面，别让没有日期的记录顶掉有日期的
        if (fact.filed or date.min) > (current.filed or date.min):
            best[fact.period_end] = fact
    return [best[k] for k in sorted(best)]


def build_series(
    spec: ConceptSpec,
    facts: list[RawFact],
    *,
    cik: int | str,
    company_name: str,
    periods: int,
    retrieved_at: datetime | None = None,
) -> tuple[MetricSeries, list[Source]] | None:
    if not facts:
        return None

    retrieved_at = retrieved_at or datetime.now(UTC)
    selected = dedupe_latest_filed(facts)[-periods:]
    if not selected:
        return None

    sources: dict[str, Source] = {}
    points: list[MetricPoint] = []
    for fact in selected:
        if fact.accession:
            url = filing_url(cik, fact.accession)
            locator = (
                f"{fact.form or 'filing'} {period_label(fact)}, XBRL {concept_name(fact.concept)}"
            )
            source_id = make_source_id(url, locator)
            if source_id not in sources:
                sources[source_id] = Source(
                    id=source_id,
                    kind=SourceKind.SEC_FILING,
                    title=f"{company_name} — {fact.form or 'SEC filing'} ({period_label(fact)})",
                    url=url,
                    locator=locator,
                    snippet=f"{spec.label} = {fact.value:,.0f} USD ({fact.concept})",
                    published_at=(
                        datetime.combine(fact.filed, datetime.min.time(), tzinfo=UTC)
                        if fact.filed
                        else None
                    ),
                    retrieved_at=retrieved_at,
                )
        value = round(fact.value / 1_000_000, 1) if spec.unit == "USD_millions" else fact.value
        points.append(MetricPoint(period=period_label(fact), value=value))

    ordered_sources = list(sources.values())
    if not ordered_sources:
        return None

    series = MetricSeries(
        key=spec.key,
        label=spec.label,
        unit=spec.unit,
        points=points,
        source_ids=[s.id for s in ordered_sources],
    )
    return series, ordered_sources


# --------------------------------------------------------------------------
# 取数与概念选择
# --------------------------------------------------------------------------


def coverage(facts: list[RawFact]) -> tuple[int, date]:
    """一组事实的"覆盖度"：不重复的期间数 + 最新期间。

    用来在多个候选概念名之间做选择。键的顺序是有意的：先比期间数量，
    数量相同再比谁更新。
    """
    if not facts:
        return (0, date.min)
    ends = {f.period_end for f in facts}
    return (len(ends), max(ends))


def select_for(spec: ConceptSpec, by_concept: dict[str, list[RawFact]]) -> list[RawFact]:
    """从已分组的事实里，挑出这个指标该用的那一组。

    两条规则，各自防一类错误：

    1. **只认候选清单里的概念名。** edgartools 的 `by_concept` 非精确模式是
       **子串匹配**（源码：`concept_lower in f.concept.lower()`），
       `by_concept("Revenue")` 会匹配到 us-gaap:CostOfRevenue——名字里含
       "revenue" 就算。不做白名单校验，营收列里会出现销售成本的数字。

    2. **在命中的候选里按覆盖度选，不是按清单顺序取第一个。** 公司会换概念名，
       老名字下可能只剩几条七年前的数据。取第一个非空结果 = 静默返回过期数据。
    """
    hits = {name: by_concept[name] for name in spec.candidates if name in by_concept}
    if not hits:
        return []
    best = max(hits, key=lambda name: coverage(hits[name]))
    return hits[best]


def _to_raw(fact: Any) -> RawFact | None:
    """edgartools 的 FinancialFact -> 我们的 RawFact。

    带维度（dimensions）的事实直接丢弃：那些是分部/地区的拆分数字，
    比如"数据中心业务的营收"。混进合并报表的时间序列里，
    你会得到一条时高时低、毫无意义的曲线。
    """
    value = getattr(fact, "numeric_value", None)
    period_end = getattr(fact, "period_end", None)
    if value is None or period_end is None:
        return None
    if getattr(fact, "dimensions", None):
        return None
    return RawFact(
        concept=str(getattr(fact, "concept", "")),
        value=float(value),
        period_start=getattr(fact, "period_start", None),
        period_end=period_end,
        fiscal_year=getattr(fact, "fiscal_year", None),
        fiscal_period=getattr(fact, "fiscal_period", None),
        form=getattr(fact, "form_type", None),
        accession=getattr(fact, "accession", None),
        filed=getattr(fact, "filing_date", None),
    )


def _company(ticker: str) -> Any:
    """拿到 edgartools 的 Company 对象，顺便完成身份声明。"""
    settings = get_settings()
    if "example.com" in settings.sec_user_agent:
        raise RuntimeError(
            "TRA_SEC_USER_AGENT 还是默认占位值，SEC 会拒绝请求。"
            "请在 apps/api/.env 里填成 '项目名 你的邮箱'。"
        )

    from edgar import Company, set_identity

    set_identity(settings.sec_user_agent)
    sec_limiter().acquire()
    return Company(ticker.strip().upper())


def group_by_concept(facts: list[RawFact]) -> dict[str, list[RawFact]]:
    """按归一化后的概念名分组。"""
    grouped: dict[str, list[RawFact]] = {}
    for fact in facts:
        grouped.setdefault(concept_name(fact.concept), []).append(fact)
    return grouped


def fetch_period_facts(company: Any, *, months: int = 3) -> list[RawFact]:
    """一次把该公司所有指定期间长度的事实取回来。

    `company.facts` 只下载一次 companyfacts，之后的 query 都是内存过滤，
    所以"一次全取再本地分组"比"每个概念查一次"更省也更可控——
    最重要的是，匹配规则由我们自己定，不受 by_concept 子串匹配的摆布。
    """
    sec_limiter().acquire()
    rows = company.facts.query().by_period_length(months).execute()
    return [r for r in (_to_raw(row) for row in rows) if r is not None]


def find_identical_series(series_list: list[MetricSeries]) -> list[tuple[str, str]]:
    """找出数值完全相同的指标对。

    为什么需要这道自检：概念名匹配错了的典型症状**不是报错，是两个指标
    变成同一列数字**——营收拿到了销售成本的值。表格照打、数字都是真的、
    只是张冠李戴，肉眼很难发现。

    两个不同的财务指标在 12 个期间上逐个相等，概率约等于零。真发生了，
    一定是取数错了。这是那种"写五行代码，省你两小时"的检查。
    """
    pairs: list[tuple[str, str]] = []
    fingerprints = {
        series.key: tuple((p.period, p.value) for p in series.points) for series in series_list
    }
    keys = list(fingerprints)
    for i, left in enumerate(keys):
        for right in keys[i + 1 :]:
            if fingerprints[left] and fingerprints[left] == fingerprints[right]:
                pairs.append((left, right))
    return pairs


# --------------------------------------------------------------------------
# 工具入口
# --------------------------------------------------------------------------


def get_financials(
    ticker: str,
    *,
    metrics: tuple[str, ...] = DEFAULT_METRICS,
    periods: int = 12,
    quarterly: bool = True,
) -> ToolResult[list[MetricSeries]]:
    """取某公司最近 N 个期间的财务指标。

    返回的是**报告契约里的 MetricSeries**，可以直接进 Report，
    不需要任何一步经过模型的手。
    """
    unknown = [m for m in metrics if m not in CONCEPTS]
    if unknown:
        return ToolResult.failure(
            f"unknown metrics: {unknown}",
            hint=f"supported metrics: {sorted(CONCEPTS)}",
        )

    try:
        company = _company(ticker)
        cik = company.cik
        name = getattr(company, "display_name", None) or ticker.upper()

        by_concept = group_by_concept(fetch_period_facts(company, months=3 if quarterly else 12))

        all_series: list[MetricSeries] = []
        all_sources: list[Source] = []
        missing: list[str] = []

        for key in metrics:
            spec = CONCEPTS[key]
            facts = select_for(spec, by_concept)
            built = build_series(spec, facts, cik=cik, company_name=name, periods=periods)
            if built is None:
                missing.append(key)
                continue
            series, sources = built
            all_series.append(series)
            all_sources.extend(sources)

        if not all_series:
            return ToolResult.failure(
                f"no XBRL data found for {ticker!r} (metrics: {list(metrics)})",
                hint=(
                    "check the ticker is a US-listed filer; foreign issuers file 20-F "
                    "and may not report these concepts quarterly"
                ),
            )

        result: ToolResult[list[MetricSeries]] = ToolResult.success(all_series, all_sources)
        notes: list[str] = []
        identical = find_identical_series(all_series)
        if identical:
            notes.append(
                f"SUSPECT: these metrics have identical values, likely a concept "
                f"mapping bug: {identical}. Do not report them as independent facts"
            )
        if missing:
            notes.append(f"no data for: {missing}")
        short = [s.key for s in all_series if len(s.points) < periods]
        if short:
            notes.append(f"fewer than {periods} periods available for: {short}")
        if notes:
            result.hint = "; ".join(notes) + ". Report the gap instead of estimating."
        return result

    except Exception as exc:
        return ToolResult.from_exception(
            exc,
            hint=f"could not load SEC data for {ticker!r}; try a different ticker or report the gap",
        )
