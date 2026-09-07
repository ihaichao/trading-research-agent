"""图上的各个节点。

每个节点都是**普通函数**：接收 state，返回要合并进 state 的字典。
没有魔法——LangGraph 负责的是调度、并发和持久化，业务逻辑还是你自己的代码。
这也是为什么 M0 手写一遍 ReAct 循环是值得的：你现在知道框架替你做了什么。
"""

from __future__ import annotations

import logging
import re

from pydantic import BaseModel, Field, ValidationError

from tra.agent.evidence import EvidencePack, build_evidence, missing_periods
from tra.agent.llm import LLMCallable, extract_json
from tra.agent.synthesize import numbers_in
from tra.graph.state import Finding, ResearcherInput, ResearchState, SubQuestion
from tra.report.schema import SectionKind
from tra.tools import calc, get_quote
from tra.tools.fundamentals import COMPANY_NAMES, CONCEPTS, get_financials

log = logging.getLogger(__name__)

MAX_SUB_QUESTIONS = 5
MAX_CLAIMS_PER_RESEARCHER = 3

MAX_SOURCES_PER_CLAIM = 3
"""一条判断最多挂几个出处。超出的直接截断——挂 9 个等于没挂。"""

MAX_CLAIM_CHARS = 400
"""一条判断的长度上限。

**一条判断必须是原子的，才可能被核对。** 一段 150 词、塞了五个论断的文字挂上
三个出处，读者无从判断哪个出处支撑哪一句——引用粒度一崩，"逐句可核查"这个
承诺就名存实亡。这不是排版偏好，是可验证性问题。
"""

# 模型会照抄证据里的引用写法，把 (from abc123) / (source abc123) 写进正文。
# 正文里的出处只该由渲染层生成的角标承担。
_INLINE_CITATION_RE = re.compile(
    r"[\(\[]\s*(?:cite|source|from)s?\s*:?\s*[0-9a-f]{6,12}"
    r"(?:\s*,\s*[0-9a-f]{6,12})*\s*[\)\]]",
    re.IGNORECASE,
)


def strip_inline_citations(text: str) -> str:
    """去掉模型写进正文的裸 source id，并把删完留下的空格收拾干净。"""
    cleaned = _INLINE_CITATION_RE.sub("", text)
    cleaned = re.sub(r"\s+([,.;:%)])", r"\1", cleaned)  # 删掉标点前多出来的空格
    return re.sub(r"\s{2,}", " ", cleaned).strip()


# plan 节点失败时的兜底。**一个 agent 不该因为规划失败就整个瘫痪**——
# 退回一份固定大纲，报告质量下降但仍然可用。
DEFAULT_PLAN = [
    SubQuestion(
        key="growth",
        section=SectionKind.SUMMARY,
        question="How has revenue growth evolved, and where are the turning points?",
    ),
    SubQuestion(
        key="profitability",
        section=SectionKind.FINANCIALS,
        question="How have margins moved, and what explains any anomalous quarter?",
    ),
    SubQuestion(
        key="risks",
        section=SectionKind.RISKS,
        question="What do the numbers themselves flag as risks or open questions?",
    ),
    SubQuestion(
        key="valuation",
        section=SectionKind.VALUATION,
        question="What can and cannot be said about valuation from this data?",
    ),
]

PLANNER_PROMPT = """You are planning an equity research report.

Given the evidence available, propose 3-5 focused sub-questions. Each will be
researched independently and in parallel, so they must not overlap.

Rules:
- Only ask what the evidence can actually answer. Do not plan around data you wish existed.
- **Researchers may not do arithmetic.** They can only quote figures that already appear in
  METRICS. Never plan a question whose answer requires computing a ratio, difference or
  multiple that is not already a listed metric — for example a P/E ratio when no share
  count is present, or "R&D as a share of sales" when that ratio is not itself a metric.
- Each sub-question maps to one report section: summary, financials, valuation, risks.
- Prefer questions about turning points, divergences and anomalies over "what are the numbers".

Return ONLY JSON:
{"sub_questions": [{"key": "short_slug", "section": "summary", "question": "..."}]}"""

RESEARCHER_PROMPT = """You are answering ONE question about a company, using only the
evidence provided. Another analyst handles the other questions; stay in your lane.

Hard rules:
- Every claim MUST cite 1-3 source ids from the SOURCES list. Never invent an id.
- Every number MUST appear verbatim in METRICS. Never compute differences or ratios
  yourself: give the two endpoint values, or describe the change in words.
- Periods may not be consecutive. Only say "consecutive" / "sequentially" / "quarter over
  quarter" when the two period labels are genuinely adjacent.
- At most 3 claims. Each claim is ONE sentence, at most 40 words. A paragraph bundling
  several assertions under shared citations cannot be checked and will be discarded.
- Say what the numbers MEAN — turning points, divergences, anomalies. Do not restate the
  table. Do not write source ids into the claim text; the source_ids field handles that.
- Other analysts cover the sibling questions listed below. Do not answer theirs.
- If the evidence cannot answer the question, say so in one claim and stop.

Return ONLY JSON:
{"claims": [{"text": "...", "source_ids": ["..."], "confidence": "high|medium|low"}]}"""


class _PlanOut(BaseModel):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class _Claim(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)
    """**这里刻意不设上限。**

    上限写进 schema，一条判断多挂一个出处就会让整个 _ResearchOut 解析失败，
    连同这个 researcher 的其他两条好发现一起丢掉——真实运行里 5 个 researcher
    因此损失了 2 个，报告连 Summary 段落都没有了。

    "挂了 4 个出处"不是错误，是超出风格上限。风格问题在循环里截断即可，
    只有正确性问题（编造 id、编造数字）才值得丢弃整条判断。
    """
    confidence: str = "medium"


class _ResearchOut(BaseModel):
    claims: list[_Claim] = Field(default_factory=list)


# --------------------------------------------------------------------------
# gather：取数。**完全确定性，不碰模型。**
# --------------------------------------------------------------------------


def gather(state: ResearchState) -> dict:
    ticker = state["ticker"]
    log.info("[gather] 取 %s 的 SEC 数据与行情…", ticker)

    financials = get_financials(ticker, metrics=tuple(CONCEPTS), periods=12)
    pack = EvidencePack(ticker=ticker, company_name=COMPANY_NAMES.get(ticker, ticker))

    if financials.ok and financials.data:
        pack.series = [*financials.data, *calc.derive_all({s.key: s for s in financials.data})]
        pack.sources = list(financials.sources)
        if financials.hint:
            pack.notes.append(financials.hint)
    else:
        pack.notes.append(f"fundamentals unavailable: {financials.error}")
    pack.company_name = COMPANY_NAMES.get(ticker, ticker)

    quote = get_quote(ticker)
    if quote.ok and quote.data:
        pack.sources.extend(quote.sources)
        pack.notes.append(
            f"latest price {quote.data.price:,.2f} {quote.data.currency} "
            f"(delayed, cite {quote.sources[0].id})"
        )
    else:
        pack.notes.append("market quote unavailable; do not state a price")

    gaps = missing_periods(pack.series[0]) if pack.series else []
    log.info(
        "[gather] %d 条指标、%d 个出处、%d 个期间缺口",
        len(pack.series),
        len(pack.sources),
        len(gaps),
    )
    return {
        "company_name": pack.company_name,
        "evidence": build_evidence(pack),
        "allowed_source_ids": sorted(pack.allowed_source_ids),
        # dump 成 dict 再进 state：checkpointer 要能把它序列化落盘
        "series": [s.model_dump(mode="json") for s in pack.series],
        "sources": [s.model_dump(mode="json") for s in pack.sources],
    }


# --------------------------------------------------------------------------
# plan：拆研究子问题
# --------------------------------------------------------------------------


def _dump_plan(plan: list[SubQuestion]) -> list[dict]:
    return [q.model_dump(mode="json") for q in plan]


def make_plan_node(llm: LLMCallable):  # type: ignore[no-untyped-def]
    def plan(state: ResearchState) -> dict:
        if not state.get("sources"):
            log.warning("[plan] 没有任何证据，跳过规划")
            return {"plan": [], "problems": ["no evidence gathered"]}

        messages = [
            {"role": "system", "content": PLANNER_PROMPT},
            {
                "role": "user",
                "content": f"RESEARCH QUESTION: {state['question']}\n\n{state['evidence']}",
            },
        ]
        try:
            parsed = _PlanOut.model_validate(extract_json(llm(messages)))
            sub_questions = parsed.sub_questions[:MAX_SUB_QUESTIONS]
        except (ValueError, ValidationError) as exc:
            log.warning("[plan] 规划失败（%s），退回默认大纲", exc)
            return {"plan": _dump_plan(DEFAULT_PLAN), "problems": [f"planner fell back: {exc}"]}

        if not sub_questions:
            log.warning("[plan] 规划为空，退回默认大纲")
            return {"plan": _dump_plan(DEFAULT_PLAN), "problems": ["planner returned nothing"]}

        log.info("[plan] %d 个子问题：%s", len(sub_questions), [q.key for q in sub_questions])
        return {"plan": _dump_plan(sub_questions)}

    return plan


# --------------------------------------------------------------------------
# research：每个子问题一个，**并行执行**
# --------------------------------------------------------------------------


def make_research_node(llm: LLMCallable):  # type: ignore[no-untyped-def]
    def research(payload: ResearcherInput) -> dict:
        sub = SubQuestion.model_validate(payload["sub_question"])
        allowed = set(payload["allowed_source_ids"])
        evidence = payload["evidence"]
        others = payload.get("siblings") or []
        siblings = (
            "OTHER ANALYSTS ARE COVERING (do not answer these): " + " | ".join(others)
            if others
            else ""
        )
        log.info("[research:%s] 开始…", sub.key)

        messages = [
            {"role": "system", "content": RESEARCHER_PROMPT},
            {
                "role": "user",
                "content": f"YOUR QUESTION: {sub.question}\n{siblings}\n\n{evidence}",
            },
        ]
        try:
            parsed = _ResearchOut.model_validate(extract_json(llm(messages)))
        except (ValueError, ValidationError) as exc:
            log.warning("[research:%s] 输出不可用：%s", sub.key, exc)
            return {"problems": [f"{sub.key}: unusable output ({exc})"]}

        findings: list[Finding] = []
        problems: list[str] = []
        for claim in parsed.claims[:MAX_CLAIMS_PER_RESEARCHER]:
            # **逐条丢弃，而不是整批打回。** 一条判断编了引用，不该让这个
            # researcher 的其他发现跟着陪葬——它们是各自独立的。
            text = strip_inline_citations(claim.text)
            if len(text) > MAX_CLAIM_CHARS:
                problems.append(
                    f"{sub.key}: dropped claim of {len(text)} chars (limit {MAX_CLAIM_CHARS})"
                )
                continue
            bad_ids = [sid for sid in claim.source_ids if sid not in allowed]
            if bad_ids:
                problems.append(f"{sub.key}: dropped claim citing unknown ids {bad_ids}")
                continue
            source_ids = claim.source_ids[:MAX_SOURCES_PER_CLAIM]
            unsupported = sorted(numbers_in(text) - numbers_in(evidence))
            if unsupported:
                problems.append(f"{sub.key}: dropped claim with unsupported numbers {unsupported}")
                continue
            findings.append(
                Finding(
                    section=sub.section,
                    text=text,
                    source_ids=source_ids,
                    confidence=claim.confidence  # type: ignore[arg-type]
                    if claim.confidence in ("high", "medium", "low")
                    else "medium",
                    dimension=sub.key,
                )
            )

        log.info("[research:%s] %d 条发现，丢弃 %d 条", sub.key, len(findings), len(problems))
        return {"findings": [f.model_dump(mode="json") for f in findings], "problems": problems}

    return research
