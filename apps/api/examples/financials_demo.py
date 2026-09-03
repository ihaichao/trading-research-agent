"""M1 的验收：完全不碰 LLM，打印一家公司的财务表。

    uv run python examples/financials_demo.py NVDA

这个脚本存在的意义是**证明数据层独立成立**。如果这里打不出正确的数字，
后面接上 agent 也只是给错误的数据套一层花哨的叙述。
数据先于智能——工具层的质量就是报告质量的上限。
"""

from __future__ import annotations

import re
import sys

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from tra.report.schema import MetricSeries
from tra.tools import calc, get_quote
from tra.tools.fundamentals import CONCEPTS, get_financials

console = Console()
PERIODS = 12

_PERIOD_RE = re.compile(r"^FY(\d{4})(?:Q(\d))?$")


def period_key(period: str) -> tuple[int, int, str]:
    """让 FY2025Q4 排在 FY2026Q1 前面。年报（无季度）排在该财年最后。"""
    match = _PERIOD_RE.match(period)
    if not match:
        return (9999, 9, period)
    year, quarter = match.group(1), match.group(2)
    return (int(year), int(quarter) if quarter else 9, period)


def format_cell(value: float | None, unit: str) -> str:
    if value is None:
        return "—"
    if unit == "percent":
        return f"{value:.1f}%"
    return f"{value:,.0f}"


def render(title: str, series_list: list[MetricSeries]) -> None:
    """期间做行、指标做列。

    反过来（期间做列）在终端里必然被截断成 FY2… ——一个季度一列，
    12 个季度就是 12 列，任何终端都放不下。
    """
    if not series_list:
        return

    periods = sorted({p.period for s in series_list for p in s.points}, key=period_key)
    periods = periods[-PERIODS:]

    table = Table(title=title, title_justify="left", header_style="bold")
    table.add_column("期间", no_wrap=True)
    for series in series_list:
        table.add_column(series.label, justify="right")

    lookup = [{p.period: p.value for p in s.points} for s in series_list]
    for period in periods:
        cells = [
            format_cell(values.get(period), series.unit)
            for values, series in zip(lookup, series_list, strict=True)
        ]
        table.add_row(period, *cells)

    console.print(table)
    console.print()


def main() -> None:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    console.rule(f"[bold]{ticker}")

    result = get_financials(ticker, metrics=tuple(CONCEPTS), periods=PERIODS)
    if not result.ok or result.data is None:
        console.print("[red]取数失败[/red]")
        console.print(escape(result.for_model()))
        raise SystemExit(1)

    reported = result.data
    derived = calc.derive_all({s.key: s for s in reported})

    render("SEC 申报数据（百万美元）", reported)
    render("派生指标（全部由 Python 计算，不经过模型）", derived)

    quote = get_quote(ticker)
    if quote.ok and quote.data:
        console.print(
            f"最新价格 {quote.data.price:,.2f} {quote.data.currency}"
            f"（{quote.data.provider}，延迟数据）\n"
        )
    else:
        console.print(f"[yellow]行情不可用[/yellow] {escape(quote.error or '')}\n")

    # markup=False：source id 形如 [1baa0b7538]，rich 会把方括号当样式标签吃掉
    console.print("[bold]出处[/bold]")
    for source in result.sources[:6]:
        console.print(f"  [{source.id}] {source.locator}", markup=False)
        console.print(f"      {source.url}", markup=False, highlight=False)
    if len(result.sources) > 6:
        console.print(f"  ...共 {len(result.sources)} 条")

    if result.hint:
        console.print(f"\n[yellow]提示[/yellow] {escape(result.hint)}")


if __name__ == "__main__":
    main()
