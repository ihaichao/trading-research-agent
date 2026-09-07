"""派生指标的计算。**所有算术都在这里，一步都不交给模型。**

为什么这是条硬规则：LLM 会把 46743/13.5 算成一个看起来很合理的错数，
而且语气毫无波动。把可计算的部分从模型手里彻底拿走，模型只负责挑指标和解读趋势。

引用的传递：派生指标的 `source_ids` 是**输入指标的 source_ids 的并集**。
毛利率算出来不产生新的出处，它的可信度完全来自营收和成本那两条的出处。
"""

from __future__ import annotations

import re

from tra.report.schema import MetricPoint, MetricSeries


def _by_period(series: MetricSeries) -> dict[str, float]:
    return {p.period: p.value for p in series.points if p.value is not None}


def _merged_source_ids(*series: MetricSeries) -> list[str]:
    """并集，保持首次出现的顺序（稳定输出便于 diff）。"""
    seen: dict[str, None] = {}
    for s in series:
        for sid in s.source_ids:
            seen.setdefault(sid, None)
    return list(seen)


def ratio(
    numerator: MetricSeries,
    denominator: MetricSeries,
    *,
    key: str,
    label: str,
    as_percent: bool = True,
) -> MetricSeries | None:
    """逐期相除。只保留两边都有值的期间。

    分母为 0 时该期的值是 None，不是 0 —— "算不出来"和"等于零"是两回事，
    混淆它们会让报告里出现一个假的 0% 毛利率。
    """
    num = _by_period(numerator)
    den = _by_period(denominator)
    periods = [p.period for p in numerator.points if p.period in num and p.period in den]
    if not periods:
        return None

    points: list[MetricPoint] = []
    for period in periods:
        d = den[period]
        value = None if d == 0 else num[period] / d * (100.0 if as_percent else 1.0)
        points.append(MetricPoint(period=period, value=None if value is None else round(value, 2)))

    return MetricSeries(
        key=key,
        label=label,
        unit="percent" if as_percent else "ratio",
        points=points,
        source_ids=_merged_source_ids(numerator, denominator),
    )


def difference(
    minuend: MetricSeries,
    subtrahend: MetricSeries,
    *,
    key: str,
    label: str,
) -> MetricSeries | None:
    """逐期相减，比如 毛利 = 营收 − 销售成本。

    两条 series 的 unit 必须一致，否则是在拿百万美元减百分比。
    """
    if minuend.unit != subtrahend.unit:
        raise ValueError(f"单位不一致：{minuend.unit} vs {subtrahend.unit}")

    a, b = _by_period(minuend), _by_period(subtrahend)
    periods = [p.period for p in minuend.points if p.period in a and p.period in b]
    if not periods:
        return None

    return MetricSeries(
        key=key,
        label=label,
        unit=minuend.unit,
        points=[MetricPoint(period=p, value=round(a[p] - b[p], 2)) for p in periods],
        source_ids=_merged_source_ids(minuend, subtrahend),
    )


_PERIOD_RE = re.compile(r"^FY(\d{4})Q([1-4])$")


def parse_period(label: str) -> tuple[int, int] | None:
    """FY2026Q2 -> (2026, 2)。不是这个格式就返回 None。"""
    match = _PERIOD_RE.match(label)
    return (int(match.group(1)), int(match.group(2))) if match else None


def shift_period(label: str, back: int) -> str | None:
    """往前推 N 个季度。FY2026Q2 往前 4 个 = FY2025Q2。"""
    parsed = parse_period(label)
    if parsed is None:
        return None
    year, quarter = parsed
    index = year * 4 + (quarter - 1) - back
    return f"FY{index // 4}Q{index % 4 + 1}"


def growth(series: MetricSeries, *, back: int, key: str, label: str) -> MetricSeries | None:
    """同比 / 环比增长率（百分比）。**按期间标签匹配，不按数组位置。**

    back=4 是同比（去年同一财季），back=1 是环比。

    为什么必须按标签匹配：SEC 的季度序列**天然缺 Q4**——大多数公司的 10-K
    只报全年、不单独报第四季度。按位置往前数 4 格，会拿去年 Q1 当同期基数，
    算出一个漂亮的假数字。这个 bug 在 NVDA 的真实数据上出现过：
    同比被算成 339%，实际是 262%。

    基期为 0 或负数时跳过该期：扭亏为盈算不出有意义的百分比，
    报"扭亏为盈"才对，报"增长 -350%"是胡说。

    期间标签不是 FYxxxxQn 时整条返回 None（比如财年未知、退化成了 ISO 日期）。
    **算不了就别算**——一条按位置硬凑的曲线比没有更糟。
    """
    if back < 1:
        return None

    values = {p.period: p.value for p in series.points}
    out: list[MetricPoint] = []
    for point in series.points:
        base_period = shift_period(point.period, back)
        if base_period is None:
            return None
        base = values.get(base_period)
        if point.value is None or base is None or base <= 0:
            continue
        out.append(MetricPoint(period=point.period, value=round((point.value / base - 1) * 100, 2)))

    if not out:
        return None

    return MetricSeries(
        key=key,
        label=label,
        unit="percent",
        points=out,
        source_ids=list(series.source_ids),
    )


def gross_margin(revenue: MetricSeries, gross_profit: MetricSeries) -> MetricSeries | None:
    return ratio(gross_profit, revenue, key="gross_margin", label="Gross margin")


def operating_margin(revenue: MetricSeries, operating_income: MetricSeries) -> MetricSeries | None:
    return ratio(operating_income, revenue, key="operating_margin", label="Operating margin")


def net_margin(revenue: MetricSeries, net_income: MetricSeries) -> MetricSeries | None:
    return ratio(net_income, revenue, key="net_margin", label="Net margin")


def revenue_growth_yoy(revenue: MetricSeries) -> MetricSeries | None:
    """季度数据的同比：拿去年同一个财季做基数。"""
    return growth(revenue, back=4, key="revenue_growth_yoy", label="Revenue growth (YoY)")


def revenue_growth_qoq(revenue: MetricSeries) -> MetricSeries | None:
    return growth(revenue, back=1, key="revenue_growth_qoq", label="Revenue growth (QoQ)")


def derive_all(series_by_key: dict[str, MetricSeries]) -> list[MetricSeries]:
    """有什么原始指标就派生什么，缺的静默跳过。

    工具层不该因为某家公司没报 R&D 就整个失败——报缺口是模型的事，
    这里只负责"能算的都算出来"。
    """
    derived: list[MetricSeries] = []
    revenue = series_by_key.get("revenue")

    # 很多公司不单独申报毛利，只报营收和销售成本。自己减出来即可——
    # 这属于"能算的都算"，不是估算：两个输入都来自申报数据，引用也一并继承。
    if "gross_profit" not in series_by_key:
        cost = series_by_key.get("cost_of_revenue")
        if revenue and cost:
            computed = difference(
                revenue, cost, key="gross_profit", label="Gross profit (computed)"
            )
            if computed:
                series_by_key = {**series_by_key, "gross_profit": computed}
                derived.append(computed)

    if revenue:
        for other_key, fn in (
            ("gross_profit", gross_margin),
            ("operating_income", operating_margin),
            ("net_income", net_margin),
        ):
            other = series_by_key.get(other_key)
            if other:
                result = fn(revenue, other)
                if result:
                    derived.append(result)

        for fn_single in (revenue_growth_yoy, revenue_growth_qoq):
            result = fn_single(revenue)
            if result:
                derived.append(result)

    return derived
