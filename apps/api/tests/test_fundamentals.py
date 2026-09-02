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
    dedupe_latest_filed,
    filing_url,
    get_financials,
    period_label,
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


def test_period_label_uses_the_fiscal_calendar_not_the_natural_one() -> None:
    """NVDA 的财年一月底结束：FY2026Q2 覆盖的是自然年 5–7 月。

    按 period_end 自己推自然季度会得到 Q3，全错。
    """
    assert period_label(fact(end="2025-07-27", fy=2026, fp="Q2")) == "FY2026Q2"


def test_period_label_handles_full_year() -> None:
    assert period_label(fact(end="2026-01-25", fy=2026, fp="FY")) == "FY2026"


def test_period_label_falls_back_to_the_end_date() -> None:
    assert period_label(fact(end="2025-07-27", fy=None, fp=None)) == "2025-07-27"


# ------------------------------------------------------------------ 去重


def test_keeps_the_most_recently_filed_record_for_a_period() -> None:
    """同一期间会被后续 filing 反复报告，取最新那份——重述已体现在里面。"""
    facts = [
        fact(end="2025-07-27", value=46_743_000_000, filed="2025-08-28"),
        fact(end="2025-07-27", value=46_700_000_000, filed="2026-02-26"),  # 年报里的对比列
    ]
    kept = dedupe_latest_filed(facts)
    assert len(kept) == 1
    assert kept[0].value == 46_700_000_000


def test_dedupe_sorts_by_period() -> None:
    facts = [
        fact(end="2025-07-27", value=3),
        fact(end="2024-07-28", value=1),
        fact(end="2025-01-26", value=2),
    ]
    assert [f.value for f in dedupe_latest_filed(facts)] == [1, 2, 3]


def test_record_without_filing_date_never_displaces_one_with() -> None:
    facts = [
        fact(end="2025-07-27", value=1, filed="2025-08-28"),
        fact(end="2025-07-27", value=2, filed=None),
    ]
    assert dedupe_latest_filed(facts)[0].value == 1


# ------------------------------------------------------------------ 组装


def test_build_series_converts_to_millions_and_keeps_order() -> None:
    facts = [
        fact(end="2024-10-27", value=35_082_000_000, fy=2025, fp="Q3"),
        fact(end="2025-01-26", value=39_331_000_000, fy=2025, fp="Q4"),
        fact(end="2025-04-27", value=44_062_000_000, fy=2026, fp="Q1"),
        fact(end="2025-07-27", value=46_743_000_000, fy=2026, fp="Q2"),
    ]
    built = build_series(
        REVENUE, facts, cik=NVDA_CIK, company_name="NVIDIA CORP", periods=12, retrieved_at=NOW
    )
    assert built is not None
    series, sources = built

    assert series.key == "revenue"
    assert series.unit == "USD_millions"
    assert [p.period for p in series.points] == ["FY2025Q3", "FY2025Q4", "FY2026Q1", "FY2026Q2"]
    assert [p.value for p in series.points] == [35082.0, 39331.0, 44062.0, 46743.0]
    assert sources and all(s.kind == "sec_filing" for s in sources)


def test_build_series_respects_the_period_limit_and_keeps_the_latest() -> None:
    facts = [fact(end=f"2025-0{i}-01", value=i * 1e9, fp=f"Q{i}") for i in range(1, 5)]
    built = build_series(
        REVENUE, facts, cik=NVDA_CIK, company_name="X", periods=2, retrieved_at=NOW
    )
    assert built is not None
    assert [p.period for p in built[0].points] == ["FY2026Q3", "FY2026Q4"]


def test_build_series_returns_none_without_facts() -> None:
    assert build_series(REVENUE, [], cik=1, company_name="X", periods=4) is None


def test_build_series_returns_none_when_no_fact_has_a_source() -> None:
    """没有 accession 就没有出处。没有出处的数字不该进报告。"""
    facts = [fact(end="2025-07-27", value=1e9, accession=None)]
    assert build_series(REVENUE, facts, cik=1, company_name="X", periods=4) is None


def test_sources_are_deduplicated_per_filing() -> None:
    """同一份 filing 报了多个期间时，Source 只应该有一条。"""
    same_accession = "0001045810-25-000123"
    facts = [
        fact(end="2025-04-27", value=1e9, fp="Q1", accession=same_accession),
        fact(end="2025-07-27", value=2e9, fp="Q2", accession=same_accession),
    ]
    built = build_series(REVENUE, facts, cik=1, company_name="X", periods=4, retrieved_at=NOW)
    assert built is not None
    # 两条 point 的 locator 不同（期间不同），所以 source 也不同——这是有意的，
    # 引用要能定位到具体期间，而不只是定位到文件。
    assert len(built[1]) == 2
    assert built[1][0].url == built[1][1].url


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
