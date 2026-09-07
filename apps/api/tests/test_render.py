"""Markdown 渲染的测试。

渲染层只有一条规则：**每句判断、每个数字都能被读者点回原文。**
引用角标不是装饰，是这份报告存在的理由。
"""

from __future__ import annotations

from datetime import UTC, datetime

from tra.report.render import citation_index, format_value, render_markdown
from tra.report.schema import (
    Claim,
    MetricPoint,
    MetricSeries,
    Report,
    Section,
    SectionKind,
    Source,
    SourceKind,
)

NOW = datetime(2026, 8, 26, tzinfo=UTC)


def build_report() -> Report:
    src_a = Source(
        id="aaaaaaaaaa",
        kind=SourceKind.SEC_FILING,
        title="NVIDIA 10-Q FY2026Q2",
        url="https://www.sec.gov/a",
        locator="10-Q FY2026Q2, XBRL Revenues",
        retrieved_at=NOW,
    )
    src_b = Source(
        id="bbbbbbbbbb",
        kind=SourceKind.MARKET_DATA,
        title="NVDA quote",
        url="https://finance.yahoo.com/quote/NVDA",
        retrieved_at=NOW,
    )
    return Report(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        question="How is revenue trending?",
        generated_at=NOW,
        data_as_of=NOW,
        sections=[
            Section(
                kind=SectionKind.SUMMARY,
                title="Summary",
                claims=[
                    Claim(text="Revenue grew for four quarters.", source_ids=["aaaaaaaaaa"]),
                    Claim(
                        text="The stock is volatile.",
                        source_ids=["bbbbbbbbbb"],
                        confidence="low",
                    ),
                ],
            ),
            Section(
                kind=SectionKind.FINANCIALS,
                title="Financial performance",
                metrics=[
                    MetricSeries(
                        key="revenue",
                        label="Total revenue",
                        unit="USD_millions",
                        points=[
                            MetricPoint(period="FY2026Q1", value=44062.0),
                            MetricPoint(period="FY2026Q2", value=46743.0),
                        ],
                        source_ids=["aaaaaaaaaa"],
                    ),
                    MetricSeries(
                        key="gross_margin",
                        label="Gross margin",
                        unit="percent",
                        points=[
                            MetricPoint(period="FY2026Q1", value=60.5),
                            MetricPoint(period="FY2026Q2", value=None),
                        ],
                        source_ids=["aaaaaaaaaa"],
                    ),
                ],
            ),
        ],
        sources=[src_a, src_b],
    )


def test_every_claim_carries_a_footnote_marker() -> None:
    markdown = render_markdown(build_report())
    assert "Revenue grew for four quarters.[^1]" in markdown
    assert "The stock is volatile.[^2]" in markdown


def test_low_confidence_is_visible_to_the_reader() -> None:
    assert "*(low confidence)*" in render_markdown(build_report())


def test_footnotes_link_back_to_the_primary_source() -> None:
    markdown = render_markdown(build_report())
    assert (
        "[^1]: [NVIDIA 10-Q FY2026Q2, 10-Q FY2026Q2, XBRL Revenues](https://www.sec.gov/a)"
        in markdown
    )


def test_citation_numbers_follow_first_appearance() -> None:
    index = citation_index(build_report())
    assert index["aaaaaaaaaa"] == 1
    assert index["bbbbbbbbbb"] == 2


def test_metric_table_has_periods_as_rows() -> None:
    """12 个季度横着排，任何载体都放不下。"""
    markdown = render_markdown(build_report())
    assert "| Period | Total revenue | Gross margin |" in markdown
    assert "| FY2026Q2 | 46,743 | n/a |" in markdown


def test_missing_values_render_as_na_not_zero() -> None:
    """ "算不出来"和"等于零"绝不能长得一样。"""
    assert format_value(None, "percent") == "n/a"
    assert format_value(0.0, "percent") == "0.0%"


def test_disclaimer_is_always_rendered() -> None:
    assert "not investment advice" in render_markdown(build_report()).lower()
