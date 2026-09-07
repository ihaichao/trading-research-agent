"""基本面工具的测试。

**不联网。** 网络那一层（`_fetch_facts` / `_company`）薄到没什么可测的；
真正容易出错的是去重、期间标签、单位换算、Source 构造——这些都是纯函数，
喂固定数据就能测死。

想跑真实的 SEC 请求：`uv run pytest -m network`。
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from tra.report.schema import Report, Section, SectionKind
from tra.tools.fundamentals import (
    CONCEPTS,
    ConceptSpec,
    RawFact,
    build_series,
    concept_name,
    coverage,
    filing_url,
    find_identical_series,
    fiscal_label,
    get_financials,
    group_by_concept,
    parse_fiscal_year_end,
    period_label,
    pick_authoritative,
    select_for,
)

NOW = datetime(2026, 8, 26, tzinfo=UTC)
REVENUE = CONCEPTS["revenue"]
NVDA_CIK = 1045810


def fact(
    *,
    end: str,
    value: float = 1.0,
    fy: int | None = 2026,
    fp: str | None = "Q2",
    filed: str | None = "2025-08-28",
    accession: str | None = "0001045810-25-000123",
    form: str | None = "10-Q",
    concept: str = "RevenueFromContractWithCustomerExcludingAssessedTax",
) -> RawFact:
    return RawFact(
        concept=concept,
        value=value,
        period_start=None,
        period_end=date.fromisoformat(end),
        fiscal_year=fy,
        fiscal_period=fp,
        form=form,
        accession=accession,
        filed=date.fromisoformat(filed) if filed else None,
    )


# ------------------------------------------------------------------ URL


def test_filing_url_strips_dashes_in_the_folder_but_keeps_them_in_the_filename() -> None:
    """EDGAR 的历史遗留：目录名去横杠，文件名留横杠。写错就是 404。"""
    url = filing_url(NVDA_CIK, "0001045810-25-000123")
    assert url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581025000123/0001045810-25-000123-index.html"
    )


def test_filing_url_drops_leading_zeros_from_cik() -> None:
    assert "/data/1045810/" in filing_url("0001045810", "0001045810-25-000123")


# ------------------------------------------------------------------ 期间标签


def test_period_label_comes_from_the_end_date_not_the_filings_fy_fp() -> None:
    """SEC 的 fy/fp 描述的是**报告这条事实的那份 filing**，不是事实自己的期间。

    一份 FY2027Q2 的 10-Q 里，去年同期的对比数字也被打上 fy=2027, fp=Q2。
    拿它当标签用，整张表会整体错一年——真实数据上发生过。
    """
    stale = fact(end="2022-10-30", fy=2024, fp="Q3")  # 来自 FY2024Q3 10-Q 的对比列
    assert period_label(stale, 1) == "FY2023Q3", "必须按 period_end 推，不能信 fy/fp"


def test_fiscal_label_follows_the_companys_own_calendar() -> None:
    """NVDA 财年一月底结束：2025-07-27 结束的季度是 FY2026Q2，不是 FY2025Q3。"""
    assert fiscal_label(date(2025, 7, 27), 1) == "FY2026Q2"
    assert fiscal_label(date(2026, 1, 25), 1) == "FY2026Q4"
    assert fiscal_label(date(2023, 4, 30), 1) == "FY2024Q1"


def test_fiscal_label_handles_a_calendar_year_company() -> None:
    assert fiscal_label(date(2025, 3, 31), 12) == "FY2025Q1"
    assert fiscal_label(date(2025, 12, 31), 12) == "FY2025Q4"


def test_fiscal_label_falls_back_to_the_date_when_the_calendar_is_unknown() -> None:
    """猜错一个月份 = 整表标签全错。日期永远不会错。"""
    assert fiscal_label(date(2025, 7, 27), None) == "2025-07-27"


def test_parse_fiscal_year_end_handles_sec_formats() -> None:
    assert parse_fiscal_year_end("0131") == 1
    assert parse_fiscal_year_end("--01-31") == 1
    assert parse_fiscal_year_end("1231") == 12
    assert parse_fiscal_year_end(None) is None
    assert parse_fiscal_year_end("nonsense") is None


# ------------------------------------------------------------------ 去重


def test_restated_period_takes_the_latest_filing() -> None:
    """数值不一致意味着重述或科目重分类，取最新的——研究要的是"现在认为当时是多少"。"""
    facts = [
        fact(end="2025-07-27", value=46_743_000_000, filed="2025-08-28"),
        fact(end="2025-07-27", value=46_700_000_000, filed="2026-02-26"),
    ]
    kept = pick_authoritative(facts)
    assert len(kept) == 1
    assert kept[0].value == 46_700_000_000


def test_unchanged_period_cites_the_filing_that_first_reported_it() -> None:
    """数值一致时取最早申报的那份。

    数字一样，但出处的可读性差很多：读者点开 FY2025Q3 的引用，应该落在
    FY2025Q3 的 10-Q 上，而不是一年后那份把它列为"去年同期"的文件里。
    """
    facts = [
        fact(end="2025-07-27", value=46_743_000_000, filed="2025-08-28", accession="A"),
        fact(end="2025-07-27", value=46_743_000_000, filed="2026-08-27", accession="B"),
    ]
    kept = pick_authoritative(facts)
    assert len(kept) == 1
    assert kept[0].accession == "A", "没有重述就该指向首次报告它的文件"


def test_periods_come_back_sorted() -> None:
    facts = [
        fact(end="2025-07-27", value=3),
        fact(end="2024-07-28", value=1),
        fact(end="2025-01-26", value=2),
    ]
    assert [f.value for f in pick_authoritative(facts)] == [1, 2, 3]


def test_record_without_filing_date_never_wins() -> None:
    facts = [
        fact(end="2025-07-27", value=1, filed="2025-08-28"),
        fact(end="2025-07-27", value=1, filed=None),
    ]
    assert pick_authoritative(facts)[0].filed is not None


# ------------------------------------------------------------------ 组装


def test_build_series_converts_to_millions_and_keeps_order() -> None:
    facts = [
        fact(end="2024-10-27", value=35_082_000_000, fy=2025, fp="Q3"),
        fact(end="2025-01-26", value=39_331_000_000, fy=2025, fp="Q4"),
        fact(end="2025-04-27", value=44_062_000_000, fy=2026, fp="Q1"),
        fact(end="2025-07-27", value=46_743_000_000, fy=2026, fp="Q2"),
    ]
    built = build_series(
        REVENUE,
        facts,
        cik=NVDA_CIK,
        company_name="NVIDIA CORP",
        periods=12,
        fiscal_year_end_month=1,
        retrieved_at=NOW,
    )
    assert built is not None
    series, sources = built

    assert series.key == "revenue"
    assert series.unit == "USD_millions"
    assert [p.period for p in series.points] == ["FY2025Q3", "FY2025Q4", "FY2026Q1", "FY2026Q2"]
    assert [p.value for p in series.points] == [35082.0, 39331.0, 44062.0, 46743.0]
    assert sources and all(s.kind == "sec_filing" for s in sources)


def test_build_series_respects_the_period_limit_and_keeps_the_latest() -> None:
    # 季度末必须真的隔三个月，否则会落进同一个财季
    ends = ["2025-04-28", "2025-07-28", "2025-10-28", "2026-01-28"]
    facts = [fact(end=end, value=(i + 1) * 1e9) for i, end in enumerate(ends)]
    built = build_series(
        REVENUE,
        facts,
        cik=NVDA_CIK,
        company_name="X",
        periods=2,
        fiscal_year_end_month=1,
        retrieved_at=NOW,
    )
    assert built is not None
    assert [p.period for p in built[0].points] == ["FY2026Q3", "FY2026Q4"]


def test_build_series_returns_none_without_facts() -> None:
    assert build_series(REVENUE, [], cik=1, company_name="X", periods=4) is None


def test_build_series_returns_none_when_no_fact_has_a_source() -> None:
    """没有 accession 就没有出处。没有出处的数字不该进报告。"""
    facts = [fact(end="2025-07-27", value=1e9, accession=None)]
    assert build_series(REVENUE, facts, cik=1, company_name="X", periods=4) is None


def test_one_source_per_filing_not_per_line_item() -> None:
    """一份 filing 一条出处，哪怕它贡献了多个期间、多个指标。

    反面教材就在真实运行里：按「filing × 概念」建出处，6 个指标 × 12 个季度
    炸出 73 条参考文献。提示词被撑爆（模型因此返回过空字符串），
    报告尾部全是指向同一个 URL 的重复链接。
    """
    same_filing = "0001045810-25-000123"
    facts = [
        fact(end="2025-04-27", value=1e9, accession=same_filing),
        fact(end="2025-07-27", value=2e9, accession=same_filing),
    ]
    built = build_series(
        REVENUE,
        facts,
        cik=1,
        company_name="X",
        periods=4,
        fiscal_year_end_month=1,
        retrieved_at=NOW,
    )
    assert built is not None
    assert len(built[1]) == 1, "同一份 filing 只该有一条出处"


def test_source_locator_uses_the_filings_own_period() -> None:
    """fy/fp 描述的是 filing 自己的期间——**这才是它们的正确用法**。

    用它给 filing 命名没问题；用它给 fact 标期间才是错的（那会整表错一年）。
    """
    facts = [fact(end="2022-10-30", fy=2024, fp="Q3", accession="0001045810-23-000227")]
    built = build_series(
        REVENUE,
        facts,
        cik=1,
        company_name="NVIDIA CORP",
        periods=4,
        fiscal_year_end_month=1,
        retrieved_at=NOW,
    )
    assert built is not None
    series, sources = built
    assert series.points[0].period == "FY2023Q3", "事实的期间按 period_end 推"
    assert sources[0].locator == "10-Q FY2024Q3", "出处按 filing 自己的期间命名"


def test_series_plugs_straight_into_a_report() -> None:
    """终极验收：工具输出能直接进 Report，中间不需要任何转换。

    这一条挂了，说明工具层和报告契约脱节了。
    """
    facts = [fact(end="2025-07-27", value=46_743_000_000)]
    built = build_series(
        REVENUE, facts, cik=NVDA_CIK, company_name="NVIDIA CORP", periods=4, retrieved_at=NOW
    )
    assert built is not None
    series, sources = built

    report = Report(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        question="revenue trend?",
        generated_at=NOW,
        data_as_of=NOW,
        sections=[Section(kind=SectionKind.FINANCIALS, title="Financials", metrics=[series])],
        sources=sources,
    )
    assert report.sections[0].metrics[0].points[0].value == 46743.0


# ------------------------------------------------------------------ 工具入口


def test_unknown_metric_fails_with_an_actionable_hint() -> None:
    result = get_financials("NVDA", metrics=("ebitda",))
    assert not result.ok
    assert "ebitda" in (result.error or "")
    assert "revenue" in (result.hint or ""), "报错要告诉模型有哪些可选项"


def test_unknown_metric_does_not_hit_the_network() -> None:
    """参数校验必须在网络调用之前——省一次请求，也省一次限速等待。"""
    result = get_financials("DEFINITELY_NOT_A_TICKER", metrics=("nope",))
    assert not result.ok
    assert "unknown metrics" in (result.error or "")


@pytest.mark.network
def test_real_sec_call_returns_nvda_revenue() -> None:
    """真实联网。需要 apps/api/.env 里配好 TRA_SEC_USER_AGENT。

    uv run pytest -m network -q
    """
    result = get_financials("NVDA", metrics=("revenue",), periods=8)
    assert result.ok, result.for_model()
    assert result.data is not None
    series = result.data[0]
    assert len(series.points) >= 4
    assert all(p.value and p.value > 0 for p in series.points)
    assert result.sources and result.sources[0].url.startswith("https://www.sec.gov/Archives/")


def _spec_sanity() -> None:
    assert isinstance(REVENUE, ConceptSpec)


# ------------------------------------------------------------------ 候选选择


def test_coverage_counts_distinct_periods_then_recency() -> None:
    few_recent = [fact(end="2026-01-25"), fact(end="2025-10-26")]
    many_old = [fact(end=f"2019-0{i}-01") for i in range(1, 5)]
    assert coverage(many_old) > coverage(few_recent), "先比期间数量"

    a = [fact(end="2026-01-25"), fact(end="2025-10-26")]
    b = [fact(end="2020-01-25"), fact(end="2019-10-26")]
    assert coverage(a) > coverage(b), "数量相同再比谁更新"


def test_coverage_ignores_duplicate_periods() -> None:
    """同一期间的多条重复记录不算多份覆盖。"""
    dupes = [fact(end="2025-07-27", filed="2025-08-28"), fact(end="2025-07-27", filed="2026-02-26")]
    assert coverage(dupes)[0] == 1


def test_select_for_prefers_coverage_over_candidate_order() -> None:
    """这条钉死了 NVDA 暴露出来的第一个 bug。

    RevenueFromContractWithCustomer... 排在候选清单第一位，但只剩两条七年前的
    数据；真正在用的概念排在后面。取"第一个非空"会静默返回过期数据。
    """
    stale_first = [fact(end="2019-07-28"), fact(end="2019-10-27")]
    live_later = [fact(end=f"2025-0{i}-01") for i in range(1, 8)]

    chosen = select_for(
        CONCEPTS["revenue"],
        {
            "RevenueFromContractWithCustomerExcludingAssessedTax": stale_first,
            "Revenues": live_later,
        },
    )
    assert chosen is live_later


def test_select_for_ignores_concepts_outside_the_candidate_list() -> None:
    """这条钉死了第二个 bug，也是更危险的那个。

    edgartools 的模糊匹配是子串匹配：by_concept("Revenue") 会命中
    us-gaap:CostOfRevenue。不做白名单校验，营收列里会出现销售成本的数字——
    表格照打、数字是真的、只是张冠李戴。
    """
    chosen = select_for(
        CONCEPTS["revenue"],
        {
            "CostOfRevenue": [fact(end=f"2025-0{i}-01") for i in range(1, 8)],
            "ContractWithCustomerLiabilityRevenueRecognized": [fact(end="2025-07-27")],
        },
    )
    assert chosen == [], "候选清单之外的概念一律不认，宁可没有数据"


def test_select_for_on_empty_input() -> None:
    assert select_for(CONCEPTS["revenue"], {}) == []


def test_concept_name_strips_the_taxonomy_prefix() -> None:
    """edgartools 的概念名带前缀，我们的候选清单写的是裸名。"""
    assert concept_name("us-gaap:CostOfRevenue") == "CostOfRevenue"
    assert concept_name("CostOfRevenue") == "CostOfRevenue"


def test_group_by_concept_normalizes_names() -> None:
    facts = [
        RawFact(
            concept="us-gaap:Revenues",
            value=1.0,
            period_start=None,
            period_end=date(2025, 7, 27),
            fiscal_year=2026,
            fiscal_period="Q2",
            form="10-Q",
            accession="0001045810-25-000123",
            filed=date(2025, 8, 28),
        )
    ]
    grouped = group_by_concept(facts)
    assert list(grouped) == ["Revenues"]


# ------------------------------------------------------------------ 自检


def test_identical_series_are_flagged() -> None:
    """概念名取错的典型症状：两个指标变成同一列数字。

    这正是 NVDA 那次的现象——营收拿到了销售成本的值，表格照打、数字都真、
    只是张冠李戴。两个不同的财务指标在多个期间上逐个相等，概率约等于零。
    """
    from tra.report.schema import MetricPoint, MetricSeries

    def make(key: str, values: list[float]) -> MetricSeries:
        return MetricSeries(
            key=key,
            label=key,
            unit="USD_millions",
            points=[MetricPoint(period=f"FY2026Q{i + 1}", value=v) for i, v in enumerate(values)],
            source_ids=["x"],
        )

    same = [make("revenue", [1.0, 2.0]), make("cost_of_revenue", [1.0, 2.0])]
    assert find_identical_series(same) == [("revenue", "cost_of_revenue")]

    different = [make("revenue", [1.0, 2.0]), make("cost_of_revenue", [1.0, 3.0])]
    assert find_identical_series(different) == []


def test_identical_check_ignores_empty_series() -> None:
    assert find_identical_series([]) == []
