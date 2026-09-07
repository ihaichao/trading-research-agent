"""Report -> Markdown。

渲染规则只有一条：**每个数字和每句判断都必须能被读者点回原文。**
"""

from __future__ import annotations

from tra.report.schema import MetricSeries, Report, Section, Source


def citation_index(report: Report) -> dict[str, int]:
    """给每个 source 分配稳定的角标序号，按正文出现顺序。"""
    index: dict[str, int] = {}
    for section in report.sections:
        ids = [sid for claim in section.claims for sid in claim.source_ids]
        ids += [sid for series in section.metrics for sid in series.source_ids]
        for sid in ids:
            index.setdefault(sid, len(index) + 1)
    for source in report.sources:
        index.setdefault(source.id, len(index) + 1)
    return index


def format_value(value: float | None, unit: str) -> str:
    if value is None:
        return "n/a"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit == "ratio":
        return f"{value:.2f}x"
    return f"{value:,.0f}"


def render_metric_table(series_list: list[MetricSeries]) -> list[str]:
    """期间做行、指标做列——12 个季度横着排没有任何载体放得下。"""
    if not series_list:
        return []
    periods: list[str] = []
    for series in series_list:
        for point in series.points:
            if point.period not in periods:
                periods.append(point.period)

    header = "| Period | " + " | ".join(s.label for s in series_list) + " |"
    divider = "|---" * (len(series_list) + 1) + "|"
    rows = [header, divider]
    lookup = [{p.period: p.value for p in s.points} for s in series_list]
    for period in periods:
        cells = [
            format_value(values.get(period), series.unit)
            for values, series in zip(lookup, series_list, strict=True)
        ]
        rows.append(f"| {period} | " + " | ".join(cells) + " |")
    return rows


def render_section(section: Section, index: dict[str, int]) -> list[str]:
    lines = [f"## {section.title}", ""]
    if section.narrative_md:
        lines += [section.narrative_md, ""]

    for claim in section.claims:
        marks = "".join(f"[^{index[sid]}]" for sid in claim.source_ids if sid in index)
        flag = " *(low confidence)*" if claim.confidence == "low" else ""
        lines.append(f"- {claim.text}{marks}{flag}")
    if section.claims:
        lines.append("")

    table = render_metric_table(section.metrics)
    if table:
        lines += [*table, ""]
    return lines


def render_footnote(source: Source, number: int) -> str:
    locator = f", {source.locator}" if source.locator else ""
    return f"[^{number}]: [{source.title}{locator}]({source.url})"


def render_markdown(report: Report) -> str:
    index = citation_index(report)
    lines = [
        f"# {report.ticker} — {report.company_name}",
        "",
        f"> {report.question}",
        "",
        f"*Generated {report.generated_at.date()} · data as of {report.data_as_of.date()}*",
        "",
    ]

    for section in report.sections:
        lines += render_section(section, index)

    lines += ["## Sources", ""]
    for source in sorted(report.sources, key=lambda s: index.get(s.id, 0)):
        lines.append(render_footnote(source, index[source.id]))

    lines += ["", "---", "", f"*{report.disclaimer}*", ""]
    return "\n".join(lines)
