"""诊断脚本：这家公司到底把数据记在哪些 XBRL 概念名下。

    uv run python examples/sec_concepts.py NVDA
    uv run python examples/sec_concepts.py NVDA revenue      # 只看名字里含 revenue 的

候选清单里的概念名是拍脑袋写的（凭对 us-gaap 的一般了解），而每家公司实际
用哪个名字只有数据知道。这个脚本按覆盖度列出真实存在的概念，
让 CONCEPTS 里的清单有据可依，而不是猜。

标了 ✓ 的表示已经在候选清单里。
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.table import Table

from tra.tools.fundamentals import (
    CONCEPTS,
    _company,
    coverage,
    fetch_period_facts,
    group_by_concept,
)

console = Console()


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    needle = sys.argv[2].lower() if len(sys.argv) > 2 else ""

    known = {name for spec in CONCEPTS.values() for name in spec.candidates}

    company = _company(ticker)
    grouped = group_by_concept(fetch_period_facts(company, months=3))

    rows = sorted(grouped.items(), key=lambda kv: coverage(kv[1]), reverse=True)
    if needle:
        rows = [(name, facts) for name, facts in rows if needle in name.lower()]

    table = Table(
        title=f"{ticker} 的季度概念（按覆盖度排序，前 40）",
        title_justify="left",
        header_style="bold",
    )
    table.add_column("在清单里", justify="center")
    table.add_column("概念名", no_wrap=False)
    table.add_column("期间数", justify="right")
    table.add_column("最新期间", justify="right")
    table.add_column("最新值", justify="right")

    for name, facts in rows[:40]:
        periods, latest = coverage(facts)
        newest = max(facts, key=lambda f: f.period_end)
        table.add_row(
            "✓" if name in known else "",
            name,
            str(periods),
            latest.isoformat(),
            f"{newest.value / 1_000_000:,.0f}M",
        )

    console.print(table)
    console.print(f"\n共 {len(grouped)} 个季度概念。用第二个参数过滤，例如 revenue / income / cost")


if __name__ == "__main__":
    main()
