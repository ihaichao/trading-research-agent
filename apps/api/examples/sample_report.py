"""生成一份样例报告，让前端在后端跑通之前就能开工。

    cd apps/api && uv run python examples/sample_report.py
    # 或在仓库根目录：make schema

写出 packages/contracts/samples/sample_report.json。数据是编的（仅用于联调），
但结构与 tra.report.schema.Report 完全一致，且通过全部校验。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from tra.report import (
    Claim,
    MetricPoint,
    MetricSeries,
    Report,
    ReportMeta,
    Section,
    SectionKind,
    Source,
    SourceKind,
    make_source_id,
)


def repo_root() -> Path:
    """向上找到仓库根目录（以 packages/contracts 存在为标志）。

    比写死 parents[N] 稳：以后目录再挪，这里不用改。
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / "packages" / "contracts").is_dir():
            return parent
    raise RuntimeError("找不到仓库根目录：期望存在 packages/contracts/")


OUT = repo_root() / "packages" / "contracts" / "samples" / "sample_report.json"

FILING_URL = "https://www.sec.gov/Archives/edgar/data/0000000000/example-10q.htm"
NEWS_URL = "https://example.com/news/fictional-company-datacenter"
QUOTE_URL = "https://finance.yahoo.com/quote/EXMP"

FILING_ID = make_source_id(FILING_URL, "10-Q FY2026Q2, Item 2")
RISK_ID = make_source_id(FILING_URL, "10-K FY2025, Item 1A, para 4")
NEWS_ID = make_source_id(NEWS_URL)
QUOTE_ID = make_source_id(QUOTE_URL, "2026-08-26 close")

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


def build() -> Report:
    sources = [
        Source(
            id=FILING_ID,
            kind=SourceKind.SEC_FILING,
            title="Example Corp — Form 10-Q, fiscal Q2 2026",
            url=FILING_URL,
            locator="10-Q FY2026Q2, Item 2 (MD&A)",
            snippet=(
                "Data center revenue was $12.4 billion, up 41% from a year ago, "
                "driven by demand for accelerated computing platforms."
            ),
            published_at=datetime(2026, 8, 5, tzinfo=UTC),
            retrieved_at=NOW,
        ),
        Source(
            id=RISK_ID,
            kind=SourceKind.SEC_FILING,
            title="Example Corp — Form 10-K, fiscal 2025",
            url=FILING_URL,
            locator="10-K FY2025, Item 1A, para 4",
            snippet=(
                "A significant portion of our revenue is concentrated among a small "
                "number of customers. The loss of any one of them could materially "
                "harm our operating results."
            ),
            published_at=datetime(2026, 2, 20, tzinfo=UTC),
            retrieved_at=NOW,
        ),
        Source(
            id=NEWS_ID,
            kind=SourceKind.WEB,
            title="Example Corp expands data center capacity (fictional)",
            url=NEWS_URL,
            snippet="The company said it will add three regions over the next year.",
            published_at=datetime(2026, 8, 18, tzinfo=UTC),
            retrieved_at=NOW,
        ),
        Source(
            id=QUOTE_ID,
            kind=SourceKind.MARKET_DATA,
            title="EXMP daily close",
            url=QUOTE_URL,
            locator="2026-08-26 close",
            snippet="EXMP closed at 184.30 USD.",
            retrieved_at=NOW,
        ),
    ]

    sections = [
        Section(
            kind=SectionKind.SUMMARY,
            title="Summary",
            claims=[
                Claim(
                    text="Data center revenue grew 41% year over year in fiscal Q2 2026.",
                    source_ids=[FILING_ID],
                    confidence="high",
                ),
                Claim(
                    text="Customer concentration remains the company's most-cited risk.",
                    source_ids=[RISK_ID],
                    confidence="high",
                ),
            ],
        ),
        Section(
            kind=SectionKind.FINANCIALS,
            title="Financial performance",
            claims=[
                Claim(
                    text="Gross margin has expanded for four consecutive quarters.",
                    source_ids=[FILING_ID],
                    confidence="medium",
                )
            ],
            metrics=[
                MetricSeries(
                    key="revenue",
                    label="Total revenue",
                    unit="USD_millions",
                    points=[
                        MetricPoint(period="FY2025Q3", value=9800.0),
                        MetricPoint(period="FY2025Q4", value=10500.0),
                        MetricPoint(period="FY2026Q1", value=11600.0),
                        MetricPoint(period="FY2026Q2", value=12400.0),
                    ],
                    source_ids=[FILING_ID],
                ),
                MetricSeries(
                    key="gross_margin",
                    label="Gross margin",
                    unit="percent",
                    points=[
                        MetricPoint(period="FY2025Q3", value=71.2),
                        MetricPoint(period="FY2025Q4", value=72.0),
                        MetricPoint(period="FY2026Q1", value=73.1),
                        MetricPoint(period="FY2026Q2", value=74.5),
                    ],
                    source_ids=[FILING_ID],
                ),
            ],
            narrative_md=(
                "Revenue growth reaccelerated in the most recent quarter while gross "
                "margin continued to expand."
            ),
        ),
        Section(
            kind=SectionKind.CATALYSTS,
            title="Recent catalysts",
            claims=[
                Claim(
                    text="The company announced three new data center regions in August 2026.",
                    source_ids=[NEWS_ID],
                    confidence="medium",
                )
            ],
        ),
        Section(
            kind=SectionKind.RISKS,
            title="Risks",
            claims=[
                Claim(
                    text="Revenue is concentrated among a small number of customers.",
                    source_ids=[RISK_ID],
                    confidence="high",
                )
            ],
        ),
        Section(
            kind=SectionKind.VALUATION,
            title="Valuation",
            claims=[
                Claim(
                    text="The stock closed at 184.30 USD on 2026-08-26.",
                    source_ids=[QUOTE_ID],
                    confidence="high",
                )
            ],
        ),
    ]

    return Report(
        ticker="EXMP",
        company_name="Example Corp (fictional)",
        question="How is Example Corp's data center business trending, and what are the risks?",
        generated_at=NOW,
        data_as_of=NOW,
        sections=sections,
        sources=sources,
        meta=ReportMeta(
            model="example-model",
            duration_seconds=142.0,
            cost_usd=0.31,
            tool_calls=17,
            research_iterations=3,
        ),
    )


def main() -> None:
    report = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({report.claim_count} claims, {len(report.sources)} sources)")


if __name__ == "__main__":
    main()
