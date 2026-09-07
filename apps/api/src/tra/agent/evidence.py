"""证据包：喂给模型的那份材料。

这是整个项目里最需要克制的一层。两条规则：

1. **只放模型需要判断的东西，不放它需要搬运的东西。**
   数字已经在 MetricSeries 里，会被程序直接装进报告。模型看到它们只是为了
   知道"发生了什么"，从而写出正确的判断句——它不负责把数字抄进报告。

2. **每条证据都带 source id。**
   模型引用时只能从这份清单里挑。挑了清单外的 id，报告构造会直接失败
   （schema 层的 no-dangling-citation 校验），而不是发出去一条假引用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tra.report.schema import MetricSeries, Source
from tra.tools.calc import parse_period, shift_period


@dataclass
class EvidencePack:
    ticker: str
    company_name: str
    series: list[MetricSeries] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """工具层给的提醒，例如"某指标没有数据"。模型必须如实转述，不能自行填补。"""

    @property
    def allowed_source_ids(self) -> set[str]:
        return {s.id for s in self.sources}


def missing_periods(series: MetricSeries) -> list[str]:
    """列出序列首尾之间缺掉的财季。

    为什么必须告诉模型：SEC 的季度序列天然缺 Q4（10-K 只报全年）。
    模型看到一串标签，会合理地假设它们相邻，于是写出
    "12 个季度连续增长"、"环比下降" 这种**基于行数而非日历**的判断。
    真实运行里这两句假话都出现过。

    缺口不说，模型就会替你把它填平——而且填得很像真的。
    """
    labels = [p.period for p in series.points]
    if len(labels) < 2 or any(parse_period(label) is None for label in labels):
        return []

    present = set(labels)
    missing: list[str] = []
    cursor = labels[0]
    for _ in range(64):  # 上限保护，别让脏数据把循环拖死
        nxt = shift_period(cursor, -1)
        if nxt is None or nxt == labels[-1]:
            break
        cursor = nxt
        if cursor not in present:
            missing.append(cursor)
    return missing


def format_series(series: MetricSeries) -> str:
    cells = []
    for point in series.points:
        if point.value is None:
            cells.append(f"{point.period}=n/a")
        elif series.unit == "percent":
            cells.append(f"{point.period}={point.value:.1f}%")
        else:
            cells.append(f"{point.period}={point.value:,.0f}")
    unit = "" if series.unit == "percent" else f" ({series.unit})"
    return f"{series.label}{unit}: " + ", ".join(cells) + f"  (from {', '.join(series.source_ids)})"


def build_evidence(pack: EvidencePack) -> str:
    """把证据包渲染成提示词里的那一段。"""
    lines = [f"COMPANY: {pack.company_name} ({pack.ticker})", "", "METRICS:"]
    lines += [f"- {format_series(s)}" for s in pack.series] or ["- (none)"]

    gaps = missing_periods(pack.series[0]) if pack.series else []
    if gaps:
        lines += [
            "",
            f"MISSING QUARTERS: {', '.join(gaps)}",
            "These fiscal quarters are NOT in the data (10-K filings report the full year, "
            "not Q4 separately). The listed periods are therefore NOT consecutive. Never write "
            '"consecutive quarters", "sequentially" or "quarter over quarter" unless the two '
            "period labels being compared are genuinely adjacent.",
            "Because no fiscal year is complete here, SEASONALITY CANNOT BE ASSESSED. Never "
            "claim that revenue is concentrated in particular fiscal quarters: the later "
            "quarters look larger because the business grew, and because Q4 is absent — "
            "not because of any seasonal pattern.",
        ]

    lines += ["", "SOURCES (only these ids may be cited):"]
    for source in pack.sources:
        locator = f" — {source.locator}" if source.locator else ""
        lines.append(f"- {source.id} [{source.kind}] {source.title}{locator}")

    if pack.notes:
        lines += ["", "DATA GAPS (state these plainly; never fill them in):"]
        lines += [f"- {note}" for note in pack.notes]

    return "\n".join(lines)
