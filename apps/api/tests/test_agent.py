"""Agent 流水线的测试。不联网、不需要 API key。

这些测试守的是这个项目最核心的一条主张：
**数字不经过模型，引用不能编造。**
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tra.agent.evidence import EvidencePack, build_evidence, format_series
from tra.agent.llm import extract_json
from tra.agent.pipeline import attach_metrics
from tra.agent.synthesize import Draft, check_citations, draft_sections, to_sections
from tra.report.schema import MetricPoint, MetricSeries, SectionKind, Source, SourceKind

NOW = datetime(2026, 8, 26, tzinfo=UTC)


def source(sid: str) -> Source:
    return Source(
        id=sid,
        kind=SourceKind.SEC_FILING,
        title=f"filing {sid}",
        url="https://www.sec.gov/x",
        locator="10-Q FY2026Q2",
        retrieved_at=NOW,
    )


def series(key: str = "revenue") -> MetricSeries:
    return MetricSeries(
        key=key,
        label="Total revenue",
        unit="USD_millions",
        points=[
            MetricPoint(period="FY2026Q1", value=44062.0),
            MetricPoint(period="FY2026Q2", value=46743.0),
        ],
        source_ids=["aaaaaaaaaa"],
    )


def pack() -> EvidencePack:
    return EvidencePack(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        series=[series()],
        sources=[source("aaaaaaaaaa"), source("bbbbbbbbbb")],
        notes=["no data for: ['rnd_expense']"],
    )


class ScriptedLLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self.replies.pop(0) if self.replies else "{}"


def good_draft(source_id: str = "aaaaaaaaaa") -> str:
    return json.dumps(
        {
            "sections": [
                {
                    "kind": "summary",
                    "title": "Summary",
                    "claims": [
                        {
                            "text": "Revenue grew for two consecutive quarters.",
                            "source_ids": [source_id],
                            "confidence": "high",
                        }
                    ],
                }
            ]
        }
    )


# ------------------------------------------------------------------ 证据包


def test_evidence_lists_only_citable_ids() -> None:
    text = build_evidence(pack())
    assert "aaaaaaaaaa" in text and "bbbbbbbbbb" in text
    assert "only these ids may be cited" in text


def test_evidence_states_data_gaps() -> None:
    """缺口必须写进证据包，否则模型会自行"补全"。"""
    text = build_evidence(pack())
    assert "DATA GAPS" in text
    assert "rnd_expense" in text
    assert "never fill them in" in text


def test_series_are_rendered_with_their_citations() -> None:
    """出处用 "(from ...)" 而不是 "[cite: ...]"。

    模型会照抄证据里的写法：用 cite 字样，它就会把 "(cite: abc123)" 写进正文，
    正文里的出处只该由渲染层的角标承担。
    """
    rendered = format_series(series())
    assert "FY2026Q2=46,743" in rendered
    assert "(from aaaaaaaaaa)" in rendered
    assert "cite" not in rendered


# ------------------------------------------------------------------ JSON 解析


def test_extract_json_survives_code_fences() -> None:
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_survives_a_chatty_prefix() -> None:
    assert extract_json('好的，这是结果：\n{"a": 1}') == {"a": 1}


def test_extract_json_raises_on_garbage() -> None:
    with pytest.raises(ValueError, match="没有返回可解析的 JSON"):
        extract_json("完全不是 JSON")


# ------------------------------------------------------------------ 引用校验


def test_fabricated_citations_are_detected() -> None:
    draft = Draft.model_validate(json.loads(good_draft("zzzzzzzzzz")))
    assert check_citations(draft, {"aaaaaaaaaa"}) == ["zzzzzzzzzz"]


def test_valid_citations_pass() -> None:
    draft = Draft.model_validate(json.loads(good_draft()))
    assert check_citations(draft, {"aaaaaaaaaa"}) == []


def test_empty_sections_are_dropped() -> None:
    draft = Draft.model_validate(
        {"sections": [{"kind": "summary", "title": "Summary", "claims": []}]}
    )
    assert to_sections(draft) == []


# ------------------------------------------------------------------ 重试


def test_first_valid_draft_is_accepted() -> None:
    llm = ScriptedLLM([good_draft()])
    sections, problems = draft_sections(pack(), "q", llm)

    assert problems == []
    assert len(llm.calls) == 1
    assert sections[0].kind == SectionKind.SUMMARY


def test_fabricated_citation_triggers_a_retry_with_the_reason() -> None:
    """编造引用不是"扣分"，是打回重写，而且要告诉它错在哪。"""
    llm = ScriptedLLM([good_draft("zzzzzzzzzz"), good_draft()])
    sections, problems = draft_sections(pack(), "q", llm)

    assert len(llm.calls) == 2
    assert "zzzzzzzzzz" in problems[0]
    assert "不存在的 source id" in problems[0]
    assert sections, "第二轮修好之后应该正常产出"

    retry_prompt = llm.calls[1][-1]["content"]
    assert "zzzzzzzzzz" in retry_prompt
    assert "不要换个 id 硬凑" in retry_prompt


def test_malformed_json_triggers_a_retry() -> None:
    llm = ScriptedLLM(["这不是 JSON", good_draft()])
    sections, problems = draft_sections(pack(), "q", llm)
    assert len(llm.calls) == 2
    assert sections
    assert "不符合约定格式" in problems[0]


def test_persistent_fabrication_yields_nothing() -> None:
    """三次都在编引用，就交白卷。

    **宁可没有报告，也不能发一份带假引用的报告。**
    """
    llm = ScriptedLLM([good_draft("zzzzzzzzzz")] * 3)
    sections, problems = draft_sections(pack(), "q", llm)

    assert sections == []
    assert len(problems) == 3


# ------------------------------------------------------------------ 数字装配


def test_metrics_are_attached_by_the_program_not_the_model() -> None:
    """模型只写句子。指标表由程序直接挂上去，模型碰不到这些数字。"""
    llm = ScriptedLLM([good_draft()])
    sections, _ = draft_sections(pack(), "q", llm)
    sections = attach_metrics(sections, pack())

    financials = [s for s in sections if s.kind == SectionKind.FINANCIALS]
    assert len(financials) == 1
    assert financials[0].metrics[0].points[1].value == 46743.0


def test_attach_metrics_reuses_an_existing_financials_section() -> None:
    from tra.report.schema import Section

    existing = [Section(kind=SectionKind.FINANCIALS, title="Financials")]
    result = attach_metrics(existing, pack())
    assert len(result) == 1
    assert result[0].metrics


# ------------------------------------------------------------------ 数字核查


def test_numbers_in_ignores_small_counts() -> None:
    """ "two quarters"、"10-Q" 满地都是，检查它们只会制造噪音。"""
    from tra.agent.synthesize import numbers_in

    assert numbers_in("grew for 4 quarters in the 10-Q") == set()
    assert "46743" in numbers_in("revenue was $46,743 million")
    assert "75.5" in numbers_in("gross margin was 75.5%")


def test_numbers_not_present_in_the_evidence_are_flagged() -> None:
    """真实案例：模型写了 "$46,743 million"，而证据包里根本没有这个数。

    那是它自己的记忆。带着引用角标的错误数字，比裸奔的错误更有欺骗性。
    """
    from tra.agent.synthesize import check_numbers

    draft = Draft.model_validate(
        {
            "sections": [
                {
                    "kind": "summary",
                    "title": "Summary",
                    "claims": [
                        {
                            "text": "Revenue was $46,743 million in the latest quarter.",
                            "source_ids": ["aaaaaaaaaa"],
                        }
                    ],
                }
            ]
        }
    )
    evidence = "Total revenue (USD_millions): FY2026Q2=96,221"
    assert check_numbers(draft, evidence) == ["46743"]


def test_numbers_present_in_the_evidence_pass() -> None:
    from tra.agent.synthesize import check_numbers

    draft = Draft.model_validate(
        {
            "sections": [
                {
                    "kind": "summary",
                    "title": "Summary",
                    "claims": [
                        {
                            "text": "Revenue reached $96,221 million.",
                            "source_ids": ["aaaaaaaaaa"],
                        }
                    ],
                }
            ]
        }
    )
    assert check_numbers(draft, "Total revenue (USD_millions): FY2026Q2=96,221") == []


def test_unsupported_number_triggers_a_retry_naming_it() -> None:
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "sections": [
                        {
                            "kind": "summary",
                            "title": "Summary",
                            "claims": [
                                {
                                    "text": "Revenue was 12,345 million.",
                                    "source_ids": ["aaaaaaaaaa"],
                                }
                            ],
                        }
                    ]
                }
            ),
            good_draft(),
        ]
    )
    sections, problems = draft_sections(pack(), "q", llm)

    assert len(llm.calls) == 2
    assert "12345" in problems[0]
    assert "证据里没有的数字" in problems[0]
    assert "记不准就别写数字" in llm.calls[1][-1]["content"]
    assert sections


def test_empty_model_response_explains_the_likely_cause() -> None:
    """空字符串在真实运行里出现过：提示词太长，输出被截断。

    报错必须指向可排查的方向，而不是一句"解析失败"。
    """
    with pytest.raises(ValueError, match="空字符串"):
        extract_json("")


def test_empty_response_is_retried() -> None:
    llm = ScriptedLLM(["", good_draft()])
    sections, problems = draft_sections(pack(), "q", llm)
    assert len(llm.calls) == 2
    assert "空字符串" in problems[0]
    assert sections


# ------------------------------------------------------------------ 期间缺口


def test_missing_quarters_are_detected() -> None:
    """SEC 的季度序列天然缺 Q4——10-K 只报全年。"""
    from tra.agent.evidence import missing_periods
    from tra.report.schema import MetricPoint, MetricSeries

    quarterly = MetricSeries(
        key="revenue",
        label="Total revenue",
        unit="USD_millions",
        points=[
            MetricPoint(period=p, value=1.0)
            for p in ["FY2025Q1", "FY2025Q2", "FY2025Q3", "FY2026Q1"]
        ],
        source_ids=["aaaaaaaaaa"],
    )
    assert missing_periods(quarterly) == ["FY2025Q4"]


def test_no_gaps_reported_for_a_contiguous_series() -> None:
    from tra.agent.evidence import missing_periods
    from tra.report.schema import MetricPoint, MetricSeries

    contiguous = MetricSeries(
        key="revenue",
        label="Total revenue",
        unit="USD_millions",
        points=[MetricPoint(period=p, value=1.0) for p in ["FY2026Q1", "FY2026Q2", "FY2026Q3"]],
        source_ids=["aaaaaaaaaa"],
    )
    assert missing_periods(contiguous) == []


def test_gaps_are_spelled_out_in_the_evidence() -> None:
    """真实运行里模型写过 "12 个季度连续增长"——表里确实 12 行，但不连续。

    缺口不说，模型就会替你把它填平，而且填得很像真的。
    """
    from tra.report.schema import MetricPoint, MetricSeries

    gappy = EvidencePack(
        ticker="NVDA",
        company_name="NVIDIA CORP",
        series=[
            MetricSeries(
                key="revenue",
                label="Total revenue",
                unit="USD_millions",
                points=[
                    MetricPoint(period=p, value=1.0)
                    for p in ["FY2025Q1", "FY2025Q2", "FY2025Q3", "FY2026Q1"]
                ],
                source_ids=["aaaaaaaaaa"],
            )
        ],
        sources=[source("aaaaaaaaaa")],
    )
    text = build_evidence(gappy)
    assert "MISSING QUARTERS: FY2025Q4" in text
    assert "NOT consecutive" in text


def test_shotgun_citations_are_rejected() -> None:
    """给一句话挂 9 个出处等于没挂——读者无从核对哪个数字来自哪一份文件。"""
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "sections": [
                        {
                            "kind": "summary",
                            "title": "Summary",
                            "claims": [
                                {
                                    "text": "Everything went up.",
                                    "source_ids": ["aaaaaaaaaa"] * 9,
                                }
                            ],
                        }
                    ]
                }
            ),
            good_draft(),
        ]
    )
    sections, problems = draft_sections(pack(), "q", llm)
    assert len(llm.calls) == 2, "超出上限应该被打回重写"
    assert "不符合约定格式" in problems[0]
    assert sections


def test_timeout_is_configured_not_left_at_the_default() -> None:
    """一次卡住的请求会让整个流程静默挂死，而调用方只看到"没反应"。"""
    from tra.config import Settings

    settings = Settings(_env_file=None)
    assert settings.llm_timeout_seconds > 0
    assert settings.llm_timeout_seconds <= 600, "超时必须是个人愿意等的数字"


# ------------------------------------------------------------------ 截断 vs 打回


def test_extra_claims_are_truncated_not_rejected() -> None:
    """**风格约束靠截断，正确性约束才打回。**

    多写两条判断不是错误，为它重跑一次模型要付几分钟和一份钱；
    而编造引用是错误，必须打回。两类约束的处理方式不该一样。
    """
    from tra.agent.synthesize import MAX_CLAIMS_PER_SECTION

    draft = Draft.model_validate(
        {
            "sections": [
                {
                    "kind": "summary",
                    "title": "Summary",
                    "claims": [
                        {"text": f"claim {i}", "source_ids": ["aaaaaaaaaa"]} for i in range(9)
                    ],
                }
            ]
        }
    )
    sections = to_sections(draft)
    assert len(sections[0].claims) == MAX_CLAIMS_PER_SECTION


def test_extra_sections_are_truncated() -> None:
    from tra.agent.synthesize import MAX_SECTIONS

    draft = Draft.model_validate(
        {
            "sections": [
                {
                    "kind": kind,
                    "title": kind,
                    "claims": [{"text": "x", "source_ids": ["aaaaaaaaaa"]}],
                }
                for kind in ["summary", "financials", "valuation", "risks"]
            ]
        }
    )
    assert len(to_sections(draft)) <= MAX_SECTIONS


def test_truncation_costs_no_extra_model_call() -> None:
    """截断发生在解析之后，不触发重试——这正是它相对"打回"的意义。"""
    llm = ScriptedLLM(
        [
            json.dumps(
                {
                    "sections": [
                        {
                            "kind": "summary",
                            "title": "Summary",
                            "claims": [
                                {"text": f"claim {i}", "source_ids": ["aaaaaaaaaa"]}
                                for i in range(9)
                            ],
                        }
                    ]
                }
            )
        ]
    )
    sections, problems = draft_sections(pack(), "q", llm)
    assert len(llm.calls) == 1, "超量不该触发重试"
    assert problems == []
    assert len(sections[0].claims) == 4
