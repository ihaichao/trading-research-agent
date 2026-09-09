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
from tra.report.schema import ClaimKind, SectionKind
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


# **期间标签不是数据点。** "FY2024Q4 缺失" 说的是"你问的哪一段"，
# 没有断言任何数值——但 numbers_in 会把 2024 当成一个三位以上的数字。
# 不区分这两者，就会把唯一合法的"我答不了"当成编造数字丢掉：季节性那道题
# 就是这样从通过变回失败的。
_FISCAL_PERIOD_RE = re.compile(
    r"\bFY\s?\d{4}(?:\s?Q[1-4])?\b|\bQ[1-4]\s?FY\s?\d{4}\b|\bfiscal\s+\d{4}\b",
    re.IGNORECASE,
)


def figures_in(text: str) -> set[str]:
    """文本里真正的**数值**——先剔掉期间标签，再交给 numbers_in。"""
    return numbers_in(_FISCAL_PERIOD_RE.sub(" ", text))


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
  Exception: the user's own question is always in scope, even when the answer is
  "the evidence does not support this" — that is a real answer, and omitting it is not.
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
- Every claim about the company MUST cite 1-3 source ids from the SOURCES list.
  Never invent an id, and never cite one that is not listed.
- The ONE exception: a claim saying the evidence cannot answer the question takes
  `"source_ids": []` — an empty list. There is nothing to cite for something that
  is absent, so do not reach for an unrelated id to fill the field. Such a claim
  must contain no figures at all.
- Every number MUST appear verbatim in METRICS. Never compute differences or ratios
  yourself: give the two endpoint values, or describe the change in words.
- Periods may not be consecutive. Only say "consecutive" / "sequentially" / "quarter over
  quarter" when the two period labels are genuinely adjacent.
- At most 3 claims. Each claim is ONE sentence, at most 40 words. A paragraph bundling
  several assertions under shared citations cannot be checked and will be discarded.
- Say what the numbers MEAN — turning points, divergences, anomalies. Do not restate the
  table. Do not write source ids into the claim text; the source_ids field handles that.
- Other analysts cover the sibling questions listed below. Do not answer theirs.
- If the evidence cannot answer the question, say so in ONE claim with empty source_ids,
  and stop. Repeat the asked-about thing by name in that same sentence — "a P/E ratio
  cannot be computed: share count is not in the evidence" — so the reader sees exactly
  what was asked and why it failed. Never simply omit the topic: silence reads as if the
  question had been answered.

Return ONLY JSON:
{"claims": [{"text": "...", "source_ids": ["..."], "confidence": "high|medium|low"}]}"""


class _PlanOut(BaseModel):
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class _Claim(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(default_factory=list)
    """**这里刻意不设上下限。**

    上限设过一次（max_length=3）：多挂一个出处就让整个 _ResearchOut 解析失败，
    连同这个 researcher 的其他好发现一起陪葬——真实运行里 5 个 researcher 损失了 2 个。

    下限也设过一次（min_length=1）：模型回答不了问题时老老实实返回了
    `source_ids: []`，整批同样被打回。**结果是这个 schema 惩罚了唯一诚实的行为**，
    三道陷阱题的"我答不了"就是这样消失的。

    出处的有无由 kind 决定（见 _classify），在循环里逐条判定；数量超标截断。
    """
    kind: ClaimKind = ClaimKind.FINDING
    confidence: str = "medium"


class _ResearchOut(BaseModel):
    """**故意收成 list[dict]。**

    claims 里只要有一条不合法，`list[_Claim]` 就会让整批解析失败。
    这个错误犯了两次了，代价一次比一次隐蔽——所以这里只保证"是一组东西"，
    每条的校验放到循环里逐条做。**校验的作用域必须和失败的代价匹配。**
    """

    claims: list[dict] = Field(default_factory=list)


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


def with_direct_question(question: str, plan: list[SubQuestion]) -> list[SubQuestion]:
    """确保用户**原话提的那个问题**一定有人正面回答。

    为什么不能只靠提示词：planner 拆子问题时会自然地把"数据回答不了"的问题
    整个丢掉——它觉得那不值得研究。于是报告通篇在讲营收趋势，
    只字不提用户问的市盈率。

    **沉默会被读成"答过了"。** 用户问了什么，报告就必须正面回应什么，
    哪怕回应是"这份数据得不出结论"。

    所以这个保证做在代码里，不依赖模型听话：永远把用户原问题放在第一位。
    """
    direct = SubQuestion(key="direct", section=SectionKind.SUMMARY, question=question)
    rest = [q for q in plan if q.key != "direct"]
    return [direct, *rest][:MAX_SUB_QUESTIONS]


def make_plan_node(llm: LLMCallable):  # type: ignore[no-untyped-def]
    def plan(state: ResearchState) -> dict:
        if not state.get("sources"):
            log.warning("[plan] 没有任何证据，跳过规划")
            # **报错必须指名道姓。** "no evidence gathered" 读者无从判断
            # 是哪个标的出了问题；带上代码，日志和评测都能直接定位。
            return {
                "plan": [],
                "problems": [f"no evidence gathered for {state['ticker']}"],
            }

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
            return {
                "plan": _dump_plan(with_direct_question(state["question"], list(DEFAULT_PLAN))),
                "problems": [f"planner fell back: {exc}"],
            }

        if not sub_questions:
            log.warning("[plan] 规划为空，退回默认大纲")
            return {
                "plan": _dump_plan(with_direct_question(state["question"], list(DEFAULT_PLAN))),
                "problems": ["planner returned nothing"],
            }

        sub_questions = with_direct_question(state["question"], sub_questions)
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
        for item in parsed.claims[:MAX_CLAIMS_PER_RESEARCHER]:
            # **逐条丢弃，而不是整批打回。** 一条判断编了引用，不该让这个
            # researcher 的其他发现跟着陪葬——它们是各自独立的。
            try:
                claim = _Claim.model_validate(item)
            except ValidationError as exc:
                problems.append(
                    f"{sub.key}: dropped unparseable claim ({exc.error_count()} errors)"
                )
                continue
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
            # 期间标签在这里也一样不算数值：一句"FY2025Q4 缺失"里的 2025
            # 不该被要求"在证据里出现过"——**它指认的恰恰是证据里没有的东西。**
            unsupported = sorted(figures_in(text) - numbers_in(evidence))
            if unsupported:
                problems.append(f"{sub.key}: dropped claim with unsupported numbers {unsupported}")
                continue
            if not source_ids:
                # **零出处的判断只能是"我答不了"，而且不许带数值。**
                # 不看模型自称的 kind——它没有动机分类正确；看的是"有没有数值"
                # 这个客观事实。任何数值都必须有出处，这一条没有例外。
                # 但期间标签不算数值：说"FY2024Q4 缺失"是在指认缺口，不是在报数。
                figures = figures_in(text)
                if figures:
                    problems.append(
                        f"{sub.key}: dropped uncited claim asserting "
                        f"{sorted(figures)} — {text[:80]}"
                    )
                    continue
                kind = ClaimKind.LIMITATION
            else:
                kind = ClaimKind.FINDING
            findings.append(
                Finding(
                    section=sub.section,
                    text=text,
                    source_ids=source_ids,
                    kind=kind,
                    confidence=claim.confidence  # type: ignore[arg-type]
                    if claim.confidence in ("high", "medium", "low")
                    else "medium",
                    dimension=sub.key,
                )
            )

        log.info("[research:%s] %d 条发现，丢弃 %d 条", sub.key, len(findings), len(problems))
        return {"findings": [f.model_dump(mode="json") for f in findings], "problems": problems}

    return research
