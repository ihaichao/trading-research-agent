"""LangGraph 编排的测试。不联网、不调真模型。

守三件事：**并行真的并行**、**reducer 真的合并**、**一条坏发现不拖垮整个 researcher**。
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from tra.agent.evidence import EvidencePack
from tra.graph import build_report, compile_graph
from tra.graph.nodes import DEFAULT_PLAN, make_plan_node, make_research_node
from tra.graph.state import Finding, SectionKind, SubQuestion
from tra.report.schema import ClaimKind, MetricPoint, MetricSeries, Source, SourceKind

NOW = datetime(2026, 9, 7, tzinfo=UTC)
GOOD_ID = "aaaaaaaaaa"

EVIDENCE = (
    "COMPANY: NVIDIA CORP (NVDA)\n\n"
    "METRICS:\n"
    "- Total revenue (USD_millions): FY2026Q1=44,062, FY2026Q2=46,743  [cite: aaaaaaaaaa]\n\n"
    "SOURCES (only these ids may be cited):\n"
    "- aaaaaaaaaa [sec_filing] NVIDIA CORP — 10-Q FY2026Q2\n"
)


def source() -> Source:
    return Source(
        id=GOOD_ID,
        kind=SourceKind.SEC_FILING,
        title="NVIDIA CORP — 10-Q FY2026Q2",
        url="https://www.sec.gov/a",
        locator="10-Q FY2026Q2",
        retrieved_at=NOW,
    )


def series() -> MetricSeries:
    return MetricSeries(
        key="revenue",
        label="Total revenue",
        unit="USD_millions",
        points=[
            MetricPoint(period="FY2026Q1", value=44062.0),
            MetricPoint(period="FY2026Q2", value=46743.0),
        ],
        source_ids=[GOOD_ID],
    )


def base_state() -> dict:
    return {
        "ticker": "NVDA",
        "question": "How is it doing?",
        "company_name": "NVIDIA CORP",
        "evidence": EVIDENCE,
        "allowed_source_ids": [GOOD_ID],
        "series": [series().model_dump(mode="json")],
        "sources": [source().model_dump(mode="json")],
        "findings": [],
        "problems": [],
    }


def claim_json(text: str, ids: list[str] | None = None) -> str:
    return json.dumps(
        {"claims": [{"text": text, "source_ids": ids or [GOOD_ID], "confidence": "high"}]}
    )


class RecordingLLM:
    """记录每次调用的线程和时间，用来证明 researcher 真的在并行。"""

    def __init__(self, reply_for) -> None:  # type: ignore[no-untyped-def]
        self.reply_for = reply_for
        self.threads: set[int] = set()
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def __call__(self, messages: list[dict[str, str]]) -> str:
        content = messages[-1]["content"]
        with self.lock:
            self.threads.add(threading.get_ident())
            self.calls.append(content)
        time.sleep(0.15)
        return self.reply_for(content)


# ------------------------------------------------------------------ plan


def test_planner_falls_back_instead_of_dying() -> None:
    """规划失败不该让整个 agent 瘫痪——退回固定大纲，质量下降但仍可用。"""
    node = make_plan_node(lambda _messages: "这不是 JSON")
    out = node(base_state())  # type: ignore[arg-type]
    plan = [SubQuestion.model_validate(q) for q in out["plan"]]

    assert plan[0].key == "direct", "兜底大纲同样保证用户原问题被回答"
    assert [q.key for q in plan[1:]] == [q.key for q in DEFAULT_PLAN][: len(plan) - 1]
    assert "fell back" in out["problems"][0]


def test_planner_without_evidence_does_not_plan() -> None:
    state = base_state() | {"sources": []}
    node = make_plan_node(lambda _messages: json.dumps({"sub_questions": []}))
    out = node(state)  # type: ignore[arg-type]
    assert out["plan"] == []


def test_planner_caps_the_number_of_sub_questions() -> None:
    """上限包含那个必然存在的 direct 子问题。"""
    reply = json.dumps(
        {
            "sub_questions": [
                {"key": f"k{i}", "section": "summary", "question": f"q{i}"} for i in range(9)
            ]
        }
    )
    node = make_plan_node(lambda _messages: reply)
    assert len(node(base_state())["plan"]) == 5  # type: ignore[arg-type]


# ------------------------------------------------------------------ researcher


def payload(key: str = "growth") -> dict:
    return {
        "ticker": "NVDA",
        "company_name": "NVIDIA CORP",
        "sub_question": SubQuestion(key=key, section=SectionKind.SUMMARY, question="q").model_dump(
            mode="json"
        ),
        "evidence": EVIDENCE,
        "allowed_source_ids": [GOOD_ID],
    }


def test_researcher_produces_findings() -> None:
    node = make_research_node(lambda _m: claim_json("Revenue rose to 46,743 in FY2026Q2."))
    out = node(payload())  # type: ignore[arg-type]
    findings = [Finding.model_validate(f) for f in out["findings"]]
    assert len(findings) == 1
    assert findings[0].dimension == "growth"
    assert findings[0].section == SectionKind.SUMMARY


def test_bad_claim_is_dropped_without_killing_the_rest() -> None:
    """一条判断编了引用，不该让同一个 researcher 的其他发现陪葬。

    M2 是整批打回重写（贵）；这里是逐条丢弃（便宜），因为每条发现相互独立。
    """
    reply = json.dumps(
        {
            "claims": [
                {"text": "Revenue rose to 46,743.", "source_ids": [GOOD_ID]},
                {"text": "Something else.", "source_ids": ["zzzzzzzzzz"]},
                {"text": "Revenue was 99,999.", "source_ids": [GOOD_ID]},
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    assert len(out["findings"]) == 1, "只有第一条是干净的"
    assert any("unknown ids" in p for p in out["problems"])
    assert any("unsupported numbers" in p for p in out["problems"])


def test_unusable_output_yields_a_problem_not_an_exception() -> None:
    node = make_research_node(lambda _m: "还是不是 JSON")
    out = node(payload())  # type: ignore[arg-type]
    assert out["findings"] == [] if "findings" in out else True
    assert "unusable output" in out["problems"][0]


# ------------------------------------------------------------------ 整图


def _graph_llm() -> RecordingLLM:
    def reply_for(content: str) -> str:
        if "RESEARCH QUESTION" in content:
            return json.dumps(
                {
                    "sub_questions": [
                        {"key": "growth", "section": "summary", "question": "growth?"},
                        {"key": "margins", "section": "financials", "question": "margins?"},
                        {"key": "risks", "section": "risks", "question": "risks?"},
                    ]
                }
            )
        return claim_json("Revenue rose to 46,743 in FY2026Q2.")

    return RecordingLLM(reply_for)


def run_graph(llm, **compile_kwargs):  # type: ignore[no-untyped-def]
    from tra.graph import main_graph

    graph = main_graph.build_graph(llm)
    graph.nodes.pop("gather")  # 取数已在 base_state 里备好，测试不联网
    graph.add_node("gather", lambda _state: {})
    app = graph.compile(**compile_kwargs)
    return app


def test_findings_from_every_researcher_are_merged() -> None:
    """reducer 的活：三个 researcher 并行写 findings，一条都不能丢。

    去掉 `Annotated[list, add]` 这里会直接报并发更新错误——
    这是 fan-out 最常踩的坑。
    """
    llm = _graph_llm()
    app = run_graph(llm)
    out = app.invoke(base_state(), {"configurable": {"thread_id": "t"}})

    findings = [Finding.model_validate(f) for f in out["findings"]]
    # 4 个而不是 3 个：用户原问题永远会被追加一个 direct researcher
    assert len(findings) == 4
    assert {f.dimension for f in findings} == {"direct", "growth", "margins", "risks"}


def test_researchers_actually_run_in_parallel() -> None:
    """并行不是"看起来并行"——用线程 id 证明它。"""
    llm = _graph_llm()
    app = run_graph(llm)
    started = time.monotonic()
    app.invoke(base_state(), {"configurable": {"thread_id": "t"}})
    elapsed = time.monotonic() - started

    assert len(llm.threads) > 1, "所有调用都在同一个线程 = 其实是串行"
    assert elapsed < 0.15 * 4, f"4 个 researcher 串行至少要 4 × 0.15s，实际 {elapsed:.2f}s"


def test_report_groups_findings_into_sections() -> None:
    llm = _graph_llm()
    app = run_graph(llm)
    out = app.invoke(base_state(), {"configurable": {"thread_id": "t"}})

    report = build_report(out)
    assert report is not None
    kinds = [s.kind for s in report.sections]
    assert SectionKind.SUMMARY in kinds
    assert SectionKind.RISKS in kinds
    financials = next(s for s in report.sections if s.kind == SectionKind.FINANCIALS)
    assert financials.metrics, "指标表由程序挂上，不经过模型"


def test_report_is_none_without_findings() -> None:
    assert build_report(base_state()) is None  # type: ignore[arg-type]


def test_can_pause_after_planning_and_resume() -> None:
    """human-in-the-loop：规划后停下，人确认大纲，再继续。

    它依赖 checkpointer——没有持久化的状态，"停下来"就等于"丢掉进度"。
    """
    llm = _graph_llm()
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        app = run_graph(llm, checkpointer=checkpointer, interrupt_after=["plan"])
        config = {"configurable": {"thread_id": "hitl"}}

        paused = app.invoke(base_state(), config)
        assert len(paused["plan"]) == 4  # 3 个规划出来的 + 1 个用户原问题
        assert not paused.get("findings"), "暂停时还没有人开始研究"

        resumed = app.invoke(None, config)
        assert len(resumed["findings"]) == 4


def test_pause_without_checkpointer_is_refused() -> None:
    """没有持久化就没有"暂停"可言，早点报错好过跑一半丢状态。"""
    from tra.graph import compile_graph as compile_it

    with pytest.raises(ValueError, match="checkpointer"):
        compile_it(lambda _m: "", pause_after_plan=True)


def test_compile_graph_smoke() -> None:
    assert compile_graph(lambda _m: "") is not None


# ------------------------------------------------------------------ 判断质量


def test_paragraph_length_claims_are_dropped() -> None:
    """**一条判断必须是原子的，才可能被核对。**

    一段塞了五个论断的文字挂三个出处，读者无从判断哪个出处支撑哪一句——
    引用粒度一崩，"逐句可核查"就名存实亡。这不是排版偏好，是可验证性问题。
    """
    from tra.graph.nodes import MAX_CLAIM_CHARS

    long_text = "Revenue rose to 46,743 in FY2026Q2. " * 20
    node = make_research_node(lambda _m: claim_json(long_text))
    out = node(payload())  # type: ignore[arg-type]

    assert out.get("findings", []) == []
    assert f"limit {MAX_CLAIM_CHARS}" in out["problems"][0]


def test_inline_source_ids_are_stripped_from_the_text() -> None:
    """模型会照抄证据里的引用写法，把裸 id 写进正文。"""
    from tra.graph.nodes import strip_inline_citations

    assert (
        strip_inline_citations("Gross margin was 75.0% (from c1a63d8899) in FY2027Q2.")
        == "Gross margin was 75.0% in FY2027Q2."
    )
    assert strip_inline_citations("Price was 230.36 (source 37b2ef210d).") == "Price was 230.36."
    assert strip_inline_citations("Plain sentence.") == "Plain sentence."


def test_computed_percentages_are_caught_regardless_of_digits() -> None:
    """两位数百分比也必须核对。

    模型最爱自己算的就是"研发占收入 33%"——口径错误和五位数金额错误一样致命，
    而且更难被读者发现。早先只查三位以上的数，这类正好从缝里漏过去。
    """
    node = make_research_node(lambda _m: claim_json("R&D was about 33% of revenue."))
    out = node(payload())  # type: ignore[arg-type]

    assert out.get("findings", []) == []
    assert "unsupported numbers" in out["problems"][0]
    assert "33" in out["problems"][0]


def test_researchers_are_told_what_the_others_cover() -> None:
    """告诉每个 researcher 别人在管什么，减少重复劳动。"""
    seen: list[str] = []

    def spy(messages: list[dict[str, str]]) -> str:
        seen.append(messages[-1]["content"])
        return claim_json("Revenue rose to 46,743 in FY2026Q2.")

    node = make_research_node(spy)
    node({**payload(), "siblings": ["what about margins?", "what about risks?"]})  # type: ignore[arg-type]

    assert "OTHER ANALYSTS ARE COVERING" in seen[0]
    assert "what about margins?" in seen[0]


# ------------------------------------------------------------------ 局部失败不牵连


def test_extra_source_ids_are_truncated_not_fatal() -> None:
    """**这条钉死了一个真实事故。**

    出处上限原本写在 pydantic schema 里，结果一条判断多挂一个出处，
    整个 _ResearchOut 解析失败，同一个 researcher 的另外两条好发现一起陪葬。
    真实运行里 5 个 researcher 因此损失 2 个，报告连 Summary 段落都没了。

    "挂了 4 个出处"不是错误，是超出风格上限——截断就行。
    """
    from tra.graph.nodes import MAX_SOURCES_PER_CLAIM

    reply = json.dumps(
        {
            "claims": [
                {"text": "Revenue reached 46,743 in FY2026Q2.", "source_ids": [GOOD_ID] * 5},
                {"text": "Revenue rose again in FY2026Q2.", "source_ids": [GOOD_ID]},
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    findings = [Finding.model_validate(f) for f in out["findings"]]
    assert len(findings) == 2, "超限的那条不该拖垮同一批里的其他发现"
    assert len(findings[0].source_ids) == MAX_SOURCES_PER_CLAIM


# ------------------------------------------------------------------ 可持久化


def test_state_holds_only_plain_data() -> None:
    """checkpointer 要把 state 序列化落盘，塞自定义类型会有反序列化告警。

    做法是边界处校验成模型、进 state 前 dump 成 dict——类型安全一点没丢。
    """
    node = make_research_node(lambda _m: claim_json("Revenue rose to 46,743 in FY2026Q2."))
    out = node(payload())  # type: ignore[arg-type]

    assert all(isinstance(f, dict) for f in out["findings"])
    json.dumps(out["findings"])  # 能 json 序列化 = 能落盘


def test_whole_state_survives_a_json_round_trip() -> None:
    llm = _graph_llm()
    app = run_graph(llm)
    out = app.invoke(base_state(), {"configurable": {"thread_id": "t"}})

    revived = json.loads(json.dumps(out))
    report = build_report(revived)
    assert report is not None, "state 过一遍 JSON 之后仍然能装配出报告"


# ------------------------------------------------------------------ 装配阶段的兜底


def _finding(section: SectionKind, text: str, dimension: str = "d") -> dict:
    return Finding(
        section=section, text=text, source_ids=[GOOD_ID], dimension=dimension
    ).model_dump(mode="json")


def test_near_duplicate_findings_are_merged_away() -> None:
    """并行 researcher 之间必然重叠——同一个财务异常会被三个人各讲一遍。

    真实运行里 Financial performance 段落堆了 8 条，其中三条在重复讲
    FY2026Q1 那次毛利率下跌。告诉它们"别人在管什么"能减少但消不掉，
    因为 planner 拆出来的问题本身就交叉。装配这一步用词集重合度兜底。
    """
    state = base_state() | {
        "findings": [
            _finding(
                SectionKind.RISKS,
                "Gross margin collapsed to 60.5% in FY2026Q1 from 74.6% in FY2025Q3.",
                "a",
            ),
            _finding(
                SectionKind.RISKS,
                "Gross margin collapsed from 74.6% in FY2025Q3 to 60.5% in FY2026Q1.",
                "b",
            ),
            _finding(SectionKind.RISKS, "R&D reached 7,054 in FY2027Q2.", "c"),
        ]
    }
    report = build_report(state)  # type: ignore[arg-type]
    assert report is not None
    risks = next(s for s in report.sections if s.kind == SectionKind.RISKS)
    assert len(risks.claims) == 2, "两条近乎相同的判断只该留一条"


def test_sections_are_capped() -> None:
    from tra.graph.main_graph import MAX_CLAIMS_PER_SECTION

    # 必须是真正不同的句子——用编号变体的话会先被去重吃掉，测不到封顶
    texts = [
        "Gross margin reached 75.0% in FY2027Q2.",
        "Cost of revenue climbed to 24,079 in FY2027Q2.",
        "Net income was 59,688 in FY2027Q2.",
        "Operating leverage improved through the period.",
        "R&D as a share of sales fell to 7.3%.",
        "Revenue growth re-accelerated to 105.8% year over year.",
        "The FY2026Q1 quarter broke the margin trend.",
        "Quarterly scale passed 96,221 for the first time.",
    ]
    state = base_state() | {"findings": [_finding(SectionKind.FINANCIALS, t, "d") for t in texts]}
    report = build_report(state)  # type: ignore[arg-type]
    assert report is not None
    financials = next(s for s in report.sections if s.kind == SectionKind.FINANCIALS)
    assert len(financials.claims) == MAX_CLAIMS_PER_SECTION


def test_genuinely_different_claims_all_survive() -> None:
    """去重不能矫枉过正——阈值太低会把不同的判断当成重复吃掉。"""
    state = base_state() | {
        "findings": [
            _finding(SectionKind.RISKS, "Gross margin fell to 60.5% in FY2026Q1.", "a"),
            _finding(SectionKind.RISKS, "R&D spending reached 7,054 in FY2027Q2.", "b"),
            _finding(SectionKind.RISKS, "Revenue growth re-accelerated to 105.8%.", "c"),
        ]
    }
    report = build_report(state)  # type: ignore[arg-type]
    assert report is not None
    risks = next(s for s in report.sections if s.kind == SectionKind.RISKS)
    assert len(risks.claims) == 3


def test_report_is_deterministic_regardless_of_completion_order() -> None:
    """并行节点的返回顺序取决于谁先跑完。

    不排序的话，**同样的输入会产出顺序不同的报告**——M5 的评测集没法比较两次运行。
    """
    findings = [
        _finding(SectionKind.RISKS, "Alpha claim about margins.", "a"),
        _finding(SectionKind.RISKS, "Beta claim about revenue growth.", "b"),
        _finding(SectionKind.RISKS, "Gamma claim about spending levels.", "c"),
    ]
    first = build_report(base_state() | {"findings": findings})  # type: ignore[arg-type]
    second = build_report(base_state() | {"findings": list(reversed(findings))})  # type: ignore[arg-type]

    assert first is not None and second is not None
    texts = lambda r: [c.text for s in r.sections for c in s.claims]  # noqa: E731
    assert texts(first) == texts(second)


def test_evidence_forbids_seasonality_claims_when_no_year_is_complete() -> None:
    """这条钉死了一句通过全部校验的假话。

    模型写过 "revenue is concentrated in fiscal Q1 and Q2, pointing to seasonality" ——
    实际上后面的季度大只是因为业务长了 16 倍，加上 Q4 全部缺失。
    没编数字、没编引用、长度合规，三道防线一道都拦不住，只能在证据里说清楚。
    """
    from tra.agent.evidence import build_evidence
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
                source_ids=[GOOD_ID],
            )
        ],
        sources=[source()],
    )
    text = build_evidence(gappy)
    assert "SEASONALITY CANNOT BE ASSESSED" in text


def test_missing_evidence_error_names_the_ticker() -> None:
    """**报错必须指名道姓。**

    "no evidence gathered" 读者无从判断是哪个标的出了问题；
    带上代码，日志和评测都能直接定位。评测集里的 unknown-ticker
    用例就是因为报错不带代码而误判失败的。
    """
    node = make_plan_node(lambda _m: "")
    out = node({"ticker": "ZZZZQQ", "question": "q", "sources": [], "evidence": ""})  # type: ignore[arg-type]
    assert out["plan"] == []
    assert "ZZZZQQ" in out["problems"][0]


# ------------------------------------------------------------------ 用户的问题不能被丢掉


def test_the_users_own_question_is_always_researched() -> None:
    """**沉默会被读成"答过了"。**

    planner 会自然地把"数据回答不了"的问题整个丢掉——它觉得那不值得研究。
    于是报告通篇讲营收趋势，只字不提用户问的市盈率，评测里四道陷阱题全挂。

    这个保证做在代码里，不依赖模型听话。
    """
    reply = json.dumps(
        {
            "sub_questions": [
                {"key": "growth", "section": "summary", "question": "how is revenue growing?"}
            ]
        }
    )
    node = make_plan_node(lambda _m: reply)
    state = base_state() | {"question": "What is the current price-to-earnings ratio?"}
    plan = [SubQuestion.model_validate(q) for q in node(state)["plan"]]  # type: ignore[arg-type]

    assert plan[0].key == "direct"
    assert plan[0].question == "What is the current price-to-earnings ratio?"
    assert any(q.key == "growth" for q in plan), "原有子问题要保留"


def test_the_direct_question_survives_a_planner_failure() -> None:
    node = make_plan_node(lambda _m: "不是 JSON")
    state = base_state() | {"question": "What guidance did management give?"}
    plan = [SubQuestion.model_validate(q) for q in node(state)["plan"]]  # type: ignore[arg-type]

    assert plan[0].question == "What guidance did management give?"


def test_the_direct_question_is_not_duplicated() -> None:
    from tra.graph.nodes import MAX_SUB_QUESTIONS, with_direct_question

    plan = with_direct_question(
        "q",
        [SubQuestion(key=f"k{i}", section=SectionKind.RISKS, question=f"q{i}") for i in range(9)],
    )
    assert len(plan) == MAX_SUB_QUESTIONS
    assert [q.key for q in plan].count("direct") == 1

    twice = with_direct_question("q", with_direct_question("q", []))
    assert [q.key for q in twice] == ["direct"]


# ------------------------------------------------------------------ "我答不了"是一等公民


def test_an_uncited_refusal_survives_and_does_not_kill_the_batch() -> None:
    """**这条钉死了第二起同类事故。**

    `_Claim.source_ids` 曾经写着 `min_length=1`。模型答不了问题时老老实实返回了
    `"source_ids": []`，整个 _ResearchOut 解析失败，这个 researcher 全军覆没——
    于是三道陷阱题的报告里，那句"我答不了"人间蒸发，读者只看到一份闭口不谈
    市盈率的报告，会以为这个问题不重要。**schema 惩罚了唯一诚实的行为。**
    """
    reply = json.dumps(
        {
            "claims": [
                {
                    "text": "A P/E ratio cannot be computed: share count is not in the evidence.",
                    "source_ids": [],
                },
                {"text": "Revenue reached 46,743 in FY2026Q2.", "source_ids": [GOOD_ID]},
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    findings = [Finding.model_validate(f) for f in out["findings"]]
    assert len(findings) == 2, "没出处的那条不该拖垮同一批里的其他发现"
    assert findings[0].kind is ClaimKind.LIMITATION
    assert findings[0].source_ids == []
    assert findings[1].kind is ClaimKind.FINDING


def test_a_refusal_may_name_the_missing_periods() -> None:
    """**期间标签不是数据点。**

    "FY2024Q4 缺失"说的是"你问的哪一段"，没有断言任何数值。上一版把
    `numbers_in` 直接套在零出处的判断上，2024 被当成一个数字，于是唯一合法的
    那句"季节性无法评估"被当成编造数字丢掉——季节性那道题因此从通过变回失败。
    """
    reply = json.dumps(
        {
            "claims": [
                {
                    "text": (
                        "Seasonality cannot be assessed: FY2024Q4 and FY2025Q4 "
                        "are absent from the evidence."
                    ),
                    "source_ids": [],
                }
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    findings = [Finding.model_validate(f) for f in out["findings"]]
    assert len(findings) == 1, out["problems"]
    assert findings[0].kind is ClaimKind.LIMITATION


def test_an_uncited_claim_asserting_numbers_is_still_dropped() -> None:
    """放宽的是"承认局限"，不是"免引用"。

    **任何数字都必须有出处，没有例外。** 分类不看模型自称什么，
    看"这句话里有没有数字"这个客观事实——模型没有动机把自己分类正确。
    """
    reply = json.dumps(
        {"claims": [{"text": "Revenue reached 46,743 in FY2026Q2.", "source_ids": []}]}
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    assert out["findings"] == []
    assert "uncited claim asserting" in out["problems"][0]
    assert "46743" in out["problems"][0], "丢弃原因要指名是哪个数字"


def test_one_malformed_claim_does_not_take_the_others_with_it() -> None:
    """claims 现在逐条校验。整批 `list[_Claim]` 已经害过两次了。"""
    reply = json.dumps(
        {
            "claims": [
                {"text": "", "source_ids": [GOOD_ID]},  # text 为空，非法
                {"text": "Revenue reached 46,743 in FY2026Q2.", "source_ids": [GOOD_ID]},
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    assert len(out["findings"]) == 1
    assert "unparseable" in out["problems"][0]


def _limitation(section: SectionKind, text: str, dimension: str = "direct") -> dict:
    return Finding(
        section=section, text=text, source_ids=[], kind=ClaimKind.LIMITATION, dimension=dimension
    ).model_dump(mode="json")


def test_a_limitation_is_not_squeezed_out_by_the_section_cap() -> None:
    """**封顶和去重是风格预算，limitation 是硬保证——预算不该吃掉保证。**

    该被挤掉的是第六条同质发现，不是那句"你问的这件事我答不了"。
    """
    texts = [
        "Gross margin reached 75.0% in FY2027Q2.",
        "Cost of revenue climbed to 24,079 in FY2027Q2.",
        "Net income was 59,688 in FY2027Q2.",
        "Operating leverage improved through the period.",
        "R&D as a share of sales fell to 7.3%.",
        "Quarterly scale passed 96,221 for the first time.",
    ]
    findings = [_finding(SectionKind.FINANCIALS, t, "d") for t in texts]
    findings.append(
        _limitation(
            SectionKind.FINANCIALS,
            "Management guidance is not in the evidence; only historical filings were retrieved.",
        )
    )
    report = build_report(base_state() | {"findings": findings})  # type: ignore[arg-type]
    assert report is not None
    financials = next(s for s in report.sections if s.kind == SectionKind.FINANCIALS)
    kinds = [c.kind for c in financials.claims]
    assert ClaimKind.LIMITATION in kinds, "limitation 被封顶吃掉了"
    assert kinds[0] is ClaimKind.LIMITATION, "读者该先看到报告答不了什么"


def test_claims_that_differ_only_in_their_numbers_are_not_duplicates() -> None:
    """**数字才是这类句子的全部信息量，词集重合度对它不敏感。**

    真实运行里这两条讲的是两个不同的区间，却因为句式几乎一样被当成重复丢了一条。
    """
    state = base_state() | {
        "findings": [
            _finding(
                SectionKind.FINANCIALS,
                "R&D as a percentage of revenue decreased from 12.7% to 7.3%.",
                "rd",
            ),
            _finding(
                SectionKind.FINANCIALS,
                "R&D as a percentage of revenue decreased from 9.1% to 7.3%.",
                "rd",
            ),
        ]
    }
    report = build_report(state)  # type: ignore[arg-type]
    assert report is not None
    financials = next(s for s in report.sections if s.kind == SectionKind.FINANCIALS)
    assert len(financials.claims) == 2


def test_naming_a_period_that_is_absent_is_not_an_unsupported_number() -> None:
    """**一句"FY2027Q4 缺失"指认的恰恰是证据里没有的东西。**

    要求它出现在证据里，等于要求"证明缺失"的句子先证明自己不缺失。
    期间标签在整条流水线上都不算数值，这条守住这个一致性。
    """
    reply = json.dumps(
        {
            "claims": [
                {
                    "text": "FY2027Q4 has not been reported yet, so the fiscal year is incomplete.",
                    "source_ids": [GOOD_ID],
                }
            ]
        }
    )
    node = make_research_node(lambda _m: reply)
    out = node(payload())  # type: ignore[arg-type]

    assert len(out["findings"]) == 1, out["problems"]
