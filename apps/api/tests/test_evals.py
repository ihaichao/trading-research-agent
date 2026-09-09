"""评测系统自己的测试。

**评测的地基必须比被测系统更可靠**，所以这些指标全是纯函数，
而且它们自己也要被测——一个算错的指标比没有指标更糟：
它会让你相信一个错误的结论，还带着数字。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tra.evals.cases import EvalCase, load_cases
from tra.evals.judge import judge_report
from tra.evals.metrics import evaluate_report
from tra.evals.runner import render_results, summarize
from tra.report.schema import Claim, Report, Section, SectionKind, Source, SourceKind

NOW = datetime(2026, 9, 7, tzinfo=UTC)
GOOD_ID = "aaaaaaaaaa"
EVIDENCE = "Total revenue (USD_millions): FY2026Q1=44,062, FY2026Q2=46,743; gross margin 60.5%"


def make_report(claims: list[str], *, section: SectionKind = SectionKind.SUMMARY) -> Report:
    return Report(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        question="q",
        generated_at=NOW,
        data_as_of=NOW,
        sections=[
            Section(
                kind=section,
                title="Summary",
                claims=[Claim(text=t, source_ids=[GOOD_ID]) for t in claims],
            )
        ],
        sources=[
            Source(
                id=GOOD_ID,
                kind=SourceKind.SEC_FILING,
                title="10-Q",
                url="https://www.sec.gov/a",
                retrieved_at=NOW,
            )
        ],
    )


def case(**kwargs) -> EvalCase:  # type: ignore[no-untyped-def]
    base = {"id": "c", "kind": "rubric", "ticker": "NVDA", "question": "q"}
    return EvalCase.model_validate(base | kwargs)


# ------------------------------------------------------------------ 数据集


def test_dataset_loads_and_covers_all_three_kinds() -> None:
    """三类用例缺一不可：只有 factual 测不出"该沉默时是否沉默"。"""
    cases = load_cases("evals/dataset.jsonl")
    kinds = {c.kind for c in cases}
    assert kinds == {"factual", "rubric", "trap"}
    assert len(cases) >= 10


def test_trap_cases_declare_what_must_be_refused() -> None:
    for c in load_cases("evals/dataset.jsonl"):
        if c.kind == "trap":
            assert c.must_refuse_about, f"{c.id} 是陷阱题却没写该拒答什么"


# ------------------------------------------------------------------ 可核查性


def test_ungrounded_numbers_are_caught() -> None:
    """这个指标必须恒为 0——不为 0 说明前面的防线漏了。"""
    report = make_report(["Revenue was 99,999 in FY2026Q2."])
    metrics = evaluate_report(case(), report, evidence=EVIDENCE)
    assert metrics.ungrounded_numbers == ["99999"]
    assert not metrics.passed


def test_grounded_numbers_pass() -> None:
    report = make_report(["Revenue reached 46,743 in FY2026Q2."])
    metrics = evaluate_report(case(), report, evidence=EVIDENCE)
    assert metrics.ungrounded_numbers == []
    assert metrics.passed


# ------------------------------------------------------------------ 合规


def test_forbidden_language_fails_the_case() -> None:
    """不给评级和目标价是合规边界，不是质量偏好——必须是硬性指标。"""
    report = make_report(["We recommend investors accumulate the shares."])
    metrics = evaluate_report(case(), report, evidence=EVIDENCE)
    assert "we recommend" in metrics.forbidden_terms
    assert not metrics.passed


# ------------------------------------------------------------------ 陷阱题


def test_seasonality_claim_fails_when_a_year_is_incomplete() -> None:
    """钉住那句通过了全部机械校验的假话。"""
    report = make_report(["Revenue shows a clear seasonal pattern across quarters."])
    metrics = evaluate_report(case(), report, evidence=EVIDENCE, has_period_gaps=True)
    assert metrics.unsupported_seasonality
    assert not metrics.passed


def test_refusal_requires_admitting_the_limit_in_the_same_claim() -> None:
    """提到了却不承认局限，算失败。"""
    spec = case(kind="trap", must_refuse_about=["P/E"])

    dodged = make_report(["The P/E ratio looks attractive versus peers."])
    assert evaluate_report(spec, dodged, evidence=EVIDENCE).failed_refusals == ["P/E"]

    honest = make_report(["A P/E ratio cannot be computed: share count is not provided."])
    assert evaluate_report(spec, honest, evidence=EVIDENCE).failed_refusals == []


def test_silence_does_not_count_as_refusal() -> None:
    """完全不提也不算通过——用户问的就是它。"""
    spec = case(kind="trap", must_refuse_about=["guidance"])
    silent = make_report(["Revenue reached 46,743 in FY2026Q2."])
    assert evaluate_report(spec, silent, evidence=EVIDENCE).failed_refusals == ["guidance"]


def test_a_clean_failure_can_still_pass_a_trap_case() -> None:
    """不存在的股票代码就该干净地失败，那也是正确行为。"""
    spec = case(kind="trap", ticker="ZZZZQQ", must_refuse_about=["ZZZZQQ"])
    metrics = evaluate_report(spec, None, error="no evidence could be gathered for 'ZZZZQQ'")
    assert metrics.passed


def test_a_crash_without_the_expected_reason_fails() -> None:
    spec = case(kind="trap", must_refuse_about=["P/E"])
    metrics = evaluate_report(spec, None, error="ConnectionError: timed out")
    assert not metrics.passed


# ------------------------------------------------------------------ 内容质量


def test_required_string_missing_fails_a_factual_case() -> None:
    spec = case(kind="factual", must_contain=["96,221"])
    report = make_report(["Revenue reached 46,743 in FY2026Q2."])
    assert evaluate_report(spec, report, evidence=EVIDENCE).missing_required == ["96,221"]


def test_duplicate_claims_are_measured() -> None:
    report = make_report(
        [
            "Gross margin fell to 60.5% in FY2026Q1 from the prior quarter.",
            "Gross margin fell from the prior quarter to 60.5% in FY2026Q1.",
        ]
    )
    metrics = evaluate_report(case(), report, evidence=EVIDENCE)
    assert metrics.duplicate_pairs == 1


def test_long_claims_are_counted_as_non_atomic() -> None:
    report = make_report(["Revenue reached 46,743 in FY2026Q2. " * 20])
    metrics = evaluate_report(case(), report, evidence=EVIDENCE)
    assert metrics.non_atomic_claims == 1
    assert metrics.max_claim_chars > 400


# ------------------------------------------------------------------ 评判


def test_judge_returns_none_instead_of_a_default_score() -> None:
    """**绝不用默认分蒙混过去。**

    "评判挂了就给 3 分"会让你在趋势图上看到一条平稳的假线。宁可这一格空着。
    """
    assert judge_report(make_report(["x"]), lambda _m: "不是 JSON") is None


def test_judge_parses_scores() -> None:
    reply = json.dumps(
        {
            "internal_consistency": 2,
            "specificity": 4,
            "insight": 3,
            "honesty": 5,
            "contradictions": ["A says structural, B says transitory"],
            "comment": "two claims disagree about the same quarter",
        }
    )
    scores = judge_report(make_report(["x"]), lambda _m: reply)
    assert scores is not None
    assert scores.internal_consistency == 2
    assert scores.mean == pytest.approx(3.5)


# ------------------------------------------------------------------ 汇总


def test_summary_splits_by_kind() -> None:
    rows = [
        {
            "kind": "trap",
            "passed": True,
            "ungrounded_numbers": [],
            "duration_seconds": 10.0,
            "claim_count": 3,
        },
        {
            "kind": "trap",
            "passed": False,
            "ungrounded_numbers": ["1"],
            "duration_seconds": 20.0,
            "claim_count": 2,
        },
        {
            "kind": "factual",
            "passed": True,
            "ungrounded_numbers": [],
            "duration_seconds": 30.0,
            "claim_count": 4,
        },
    ]
    summary = summarize(rows)
    assert summary["by_kind"]["trap"] == {"passed": 1, "total": 2}
    assert summary["ungrounded_number_cases"] == 1
    assert summary["median_duration_seconds"] == 20.0


def test_results_render_as_readable_markdown() -> None:
    rows = [
        {
            "case_id": "nvda-seasonality-trap",
            "kind": "trap",
            "passed": False,
            "claim_count": 3,
            "duration_seconds": 12.0,
            "ungrounded_numbers": [],
            "missing_required": [],
            "failed_refusals": ["seasonal"],
            "forbidden_terms": [],
            "unsupported_seasonality": True,
            "duplicate_pairs": 0,
        }
    ]
    text = render_results(
        {"generated_at": NOW.isoformat(), "cases": 1, "summary": summarize(rows), "rows": rows}
    )
    assert "nvda-seasonality-trap" in text
    assert "未承认局限" in text
    assert "无依据的季节性结论" in text


def test_to_row_includes_the_pass_verdict() -> None:
    """`passed` 是 @property，`asdict()` 抓不到它。

    这个 bug 真实存在过：单元测试里的行是手写的，所以没暴露，
    直到跑一遍完整链路才崩。**测试用例的构造方式和生产代码不一致时，
    测试就不再是防线。**
    """
    metrics = evaluate_report(case(), make_report(["Revenue reached 46,743."]), evidence=EVIDENCE)
    row = metrics.to_row()
    assert "passed" in row
    assert row["passed"] is True
    json.dumps(row)  # 结果要能写进 json


def test_summary_accepts_rows_built_the_way_the_runner_builds_them() -> None:
    """用和生产代码同样的方式造行，而不是手写字典。"""
    rows = [
        evaluate_report(
            case(), make_report(["Revenue reached 46,743."]), evidence=EVIDENCE
        ).to_row()
        | {"kind": "factual", "ticker": "NVDA"},
        evaluate_report(case(), make_report(["Revenue was 99,999."]), evidence=EVIDENCE).to_row()
        | {"kind": "factual", "ticker": "NVDA"},
    ]
    summary = summarize(rows)
    assert summary["by_kind"]["factual"] == {"passed": 1, "total": 2}
    assert summary["ungrounded_number_cases"] == 1
    render_results({"generated_at": NOW.isoformat(), "cases": 2, "summary": summary, "rows": rows})


# ------------------------------------------------------------------ 评测器自己的 bug


def test_correct_refusal_is_not_counted_as_a_seasonality_violation() -> None:
    """**评测器的第一批失败，通常是评测器自己的。**

    第一版按关键词全文匹配，结果把「季节性无法判断」这种正确拒答判成了违规——
    那句话里当然有 "seasonality" 这个词。基线跑出来 8/12，其中三个失败
    都是这类误判。
    """
    honest = make_report(
        ["Seasonality cannot be assessed because no fiscal year is complete in the data."]
    )
    metrics = evaluate_report(case(), honest, evidence=EVIDENCE, has_period_gaps=True)
    assert not metrics.unsupported_seasonality
    assert metrics.passed


def test_actual_seasonality_assertion_still_fails() -> None:
    """放宽之后不能把真违规也放过去。"""
    bogus = make_report(["Revenue is concentrated in Q1 and Q2, a clear seasonal pattern."])
    metrics = evaluate_report(case(), bogus, evidence=EVIDENCE, has_period_gaps=True)
    assert metrics.unsupported_seasonality
    assert not metrics.passed


def test_a_refusal_in_one_claim_does_not_excuse_an_assertion_in_another() -> None:
    """逐条判定，不是全文判定——一句诚实不能替另一句背书。"""
    mixed = make_report(
        [
            "Seasonality cannot be assessed from this data.",
            "Revenue is concentrated in the second half, a seasonal pattern.",
        ]
    )
    assert evaluate_report(
        case(), mixed, evidence=EVIDENCE, has_period_gaps=True
    ).unsupported_seasonality


def test_synonyms_in_one_requirement_count_as_one() -> None:
    """'P/E|price-to-earnings' 是**一个**要求，不是两个。

    第一版把同义词写成两项，报告只用了缩写就被判失败——又一个评测器 bug。
    """
    spec = case(kind="trap", must_refuse_about=["P/E|price-to-earnings"])
    report = make_report(["A P/E ratio cannot be computed: share count is not provided."])
    assert evaluate_report(spec, report, evidence=EVIDENCE).failed_refusals == []


def test_synonym_requirement_still_fails_when_nothing_matches() -> None:
    spec = case(kind="trap", must_refuse_about=["P/E|price-to-earnings"])
    report = make_report(["Revenue reached 46,743 in FY2026Q2."])
    assert evaluate_report(spec, report, evidence=EVIDENCE).failed_refusals == [
        "P/E|price-to-earnings"
    ]


def test_dataset_synonyms_are_written_as_one_entry() -> None:
    """数据集里同义词必须合并，否则会重现那个误判。"""
    by_id = {c.id: c for c in load_cases("evals/dataset.jsonl")}
    assert by_id["nvda-pe-trap"].must_refuse_about == ["P/E|price-to-earnings|earnings multiple"]


def test_results_page_shows_why_claims_were_dropped() -> None:
    """**只记「丢了 2 条」等于没记。**

    三道陷阱题当时都显示 `dropped_claims: 2`，但无法判断被丢的是不是那条
    「我答不了」——而那是唯一要紧的信息。
    """
    rows = [
        {
            "case_id": "nvda-pe-trap",
            "kind": "trap",
            "passed": False,
            "claim_count": 7,
            "duration_seconds": 5.0,
            "ungrounded_numbers": [],
            "missing_required": [],
            "failed_refusals": ["P/E"],
            "forbidden_terms": [],
            "unsupported_seasonality": False,
            "duplicate_pairs": 0,
            "drop_reasons": ["direct: dropped claim citing unknown ids ['n/a']"],
        }
    ]
    text = render_results(
        {"generated_at": NOW.isoformat(), "cases": 1, "summary": summarize(rows), "rows": rows}
    )
    assert "被丢弃的判断" in text
    assert "citing unknown ids" in text
