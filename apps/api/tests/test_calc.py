"""派生指标计算的测试。全是纯函数，不联网。

这些测试值得写得比平常更细：**报告里每一个百分比都出自这里**，
算错一个小数点，上层所有的引用机制都白搭。
"""

from __future__ import annotations

import pytest

from tra.report.schema import MetricPoint, MetricSeries
from tra.tools import calc


def series(
    key: str, values: dict[str, float | None], *, unit: str = "USD_millions", sources=("s1",)
) -> MetricSeries:  # type: ignore[no-untyped-def]
    return MetricSeries(
        key=key,
        label=key,
        unit=unit,  # type: ignore[arg-type]
        points=[MetricPoint(period=p, value=v) for p, v in values.items()],
        source_ids=list(sources),
    )


REVENUE = series(
    "revenue",
    {"FY2025Q3": 35082.0, "FY2025Q4": 39331.0, "FY2026Q1": 44062.0, "FY2026Q2": 46743.0},
    sources=("sec_a",),
)
GROSS_PROFIT = series(
    "gross_profit",
    {"FY2025Q3": 26156.0, "FY2025Q4": 29342.0, "FY2026Q1": 26671.0, "FY2026Q2": 33865.0},
    sources=("sec_b",),
)


# ------------------------------------------------------------------ ratio


def test_gross_margin_is_a_percentage() -> None:
    result = calc.gross_margin(REVENUE, GROSS_PROFIT)
    assert result is not None
    assert result.unit == "percent"
    assert result.points[0].value == pytest.approx(74.56, abs=0.01)
    assert result.points[-1].value == pytest.approx(72.45, abs=0.01)


def test_derived_series_inherits_both_inputs_sources() -> None:
    """派生指标不产生新出处，它的可信度来自输入。"""
    result = calc.gross_margin(REVENUE, GROSS_PROFIT)
    assert result is not None
    assert set(result.source_ids) == {"sec_a", "sec_b"}


def test_only_periods_present_in_both_series_are_kept() -> None:
    partial = series("gross_profit", {"FY2026Q1": 26671.0, "FY2026Q2": 33865.0})
    result = calc.gross_margin(REVENUE, partial)
    assert result is not None
    assert [p.period for p in result.points] == ["FY2026Q1", "FY2026Q2"]


def test_zero_denominator_yields_none_not_zero() -> None:
    """ "算不出来"和"等于零"是两回事。混淆它们会让报告出现假的 0% 毛利率。"""
    revenue = series("revenue", {"FY2026Q1": 0.0})
    profit = series("gross_profit", {"FY2026Q1": 100.0})
    result = calc.ratio(profit, revenue, key="m", label="m")
    assert result is not None
    assert result.points[0].value is None


def test_ratio_returns_none_when_nothing_overlaps() -> None:
    a = series("a", {"FY2026Q1": 1.0})
    b = series("b", {"FY2025Q1": 1.0})
    assert calc.ratio(a, b, key="x", label="x") is None


def test_missing_values_are_skipped() -> None:
    revenue = series("revenue", {"FY2026Q1": 100.0, "FY2026Q2": None})
    profit = series("gross_profit", {"FY2026Q1": 70.0, "FY2026Q2": 80.0})
    result = calc.gross_margin(revenue, profit)
    assert result is not None
    assert [p.period for p in result.points] == ["FY2026Q1"]


# ------------------------------------------------------------------ difference


def test_difference_computes_gross_profit() -> None:
    cost = series("cost_of_revenue", {"FY2026Q2": 12878.0})
    revenue = series("revenue", {"FY2026Q2": 46743.0})
    result = calc.difference(revenue, cost, key="gross_profit", label="Gross profit")
    assert result is not None
    assert result.points[0].value == pytest.approx(33865.0)
    assert result.unit == "USD_millions"


def test_difference_refuses_mismatched_units() -> None:
    """拿百万美元减百分比是无声的灾难，必须炸在这里。"""
    a = series("a", {"FY2026Q2": 1.0}, unit="USD_millions")
    b = series("b", {"FY2026Q2": 1.0}, unit="percent")
    with pytest.raises(ValueError, match="单位不一致"):
        calc.difference(a, b, key="x", label="x")


# ------------------------------------------------------------------ growth


def test_yoy_growth_uses_a_four_quarter_lag() -> None:
    revenue = series(
        "revenue",
        {
            "FY2025Q1": 100.0,
            "FY2025Q2": 110.0,
            "FY2025Q3": 120.0,
            "FY2025Q4": 130.0,
            "FY2026Q1": 150.0,
        },
    )
    result = calc.revenue_growth_yoy(revenue)
    assert result is not None
    assert [p.period for p in result.points] == ["FY2026Q1"]
    assert result.points[0].value == pytest.approx(50.0)


def test_qoq_growth_uses_a_one_quarter_lag() -> None:
    revenue = series("revenue", {"FY2026Q1": 100.0, "FY2026Q2": 125.0})
    result = calc.revenue_growth_qoq(revenue)
    assert result is not None
    assert result.points[0].value == pytest.approx(25.0)


def test_growth_needs_more_points_than_the_lag() -> None:
    assert calc.revenue_growth_yoy(series("revenue", {"FY2026Q1": 1.0})) is None


def test_growth_from_a_non_positive_base_is_skipped() -> None:
    """扭亏为盈算不出有意义的百分比。报"增长 -350%" 是胡说。"""
    income = series("net_income", {"FY2026Q1": -50.0, "FY2026Q2": 125.0, "FY2026Q3": 250.0})
    result = calc.growth(income, back=1, key="g", label="g")
    assert result is not None
    assert [p.period for p in result.points] == ["FY2026Q3"], "基期为负的那期直接不给"
    assert result.points[0].value == pytest.approx(100.0)


def test_growth_series_with_nothing_computable_is_dropped() -> None:
    """一条算不出任何值的 series 进报告只会制造噪音，不如不给。"""
    income = series("net_income", {"FY2026Q1": -50.0, "FY2026Q2": 125.0})
    assert calc.growth(income, back=1, key="g", label="g") is None


def test_growth_returns_none_when_labels_are_not_fiscal_quarters() -> None:
    """财年未知时期间标签退化成 ISO 日期，这时算不了同比——算不了就别算。"""
    by_date = series("revenue", {"2025-04-27": 100.0, "2025-07-27": 125.0})
    assert calc.growth(by_date, back=1, key="g", label="g") is None


def test_shift_period_crosses_the_year_boundary() -> None:
    assert calc.shift_period("FY2026Q2", 4) == "FY2025Q2"
    assert calc.shift_period("FY2026Q1", 1) == "FY2025Q4"
    assert calc.shift_period("2025-07-27", 1) is None


def test_missing_q4_does_not_corrupt_the_year_over_year_number() -> None:
    """这条钉死了在 NVDA 真实数据上出现过的 bug。

    SEC 的季度序列天然缺 Q4（10-K 只报全年）。按数组位置往前数 4 格，
    会拿去年 Q1 当同期基数——NVDA 的同比因此被算成 339%，实际是 262%。
    """
    revenue = series(
        "revenue",
        {
            "FY2024Q1": 7192.0,
            "FY2024Q2": 13507.0,
            "FY2024Q3": 18120.0,
            # FY2024Q4 缺席，和真实数据一样
            "FY2025Q1": 26044.0,
        },
    )
    result = calc.revenue_growth_yoy(revenue)
    assert result is not None
    assert [p.period for p in result.points] == ["FY2025Q1"]
    assert result.points[0].value == pytest.approx(262.12, abs=0.01), "按位置数会得到 339%"


# ------------------------------------------------------------------ derive_all


def test_derive_all_computes_what_it_can() -> None:
    derived = calc.derive_all({"revenue": REVENUE, "gross_profit": GROSS_PROFIT})
    keys = {s.key for s in derived}
    assert "gross_margin" in keys
    assert "revenue_growth_qoq" in keys
    assert "operating_margin" not in keys, "没有营业利润就不该凭空派生"


def test_derive_all_without_revenue_derives_nothing() -> None:
    assert calc.derive_all({"gross_profit": GROSS_PROFIT}) == []


def test_derive_all_is_safe_on_empty_input() -> None:
    assert calc.derive_all({}) == []


# ------------------------------------------------------------------ 毛利推算


def test_gross_profit_is_computed_when_not_reported() -> None:
    """很多公司不单独申报毛利，只报营收和销售成本。

    自己减出来不算估算：两个输入都来自申报数据，引用也一并继承。
    """
    cost = series("cost_of_revenue", {"FY2026Q1": 17391.0, "FY2026Q2": 12878.0}, sources=("sec_c",))
    derived = calc.derive_all({"revenue": REVENUE, "cost_of_revenue": cost})

    computed = next(s for s in derived if s.key == "gross_profit")
    assert computed.points[0].value == pytest.approx(44062.0 - 17391.0)
    assert set(computed.source_ids) == {"sec_a", "sec_c"}, "引用必须继承两个输入"

    assert any(s.key == "gross_margin" for s in derived), "推算出的毛利要能继续算毛利率"


def test_reported_gross_profit_wins_over_the_computed_one() -> None:
    """公司自己报了毛利就用它的——申报数字优先于我们的推算。"""
    cost = series("cost_of_revenue", {"FY2026Q1": 1.0, "FY2026Q2": 2.0})
    derived = calc.derive_all(
        {"revenue": REVENUE, "cost_of_revenue": cost, "gross_profit": GROSS_PROFIT}
    )
    assert not any(s.key == "gross_profit" for s in derived), "已申报就不该再推算一份"
