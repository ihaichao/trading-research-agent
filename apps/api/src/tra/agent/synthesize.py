"""让模型起草报告的文字部分，并把它约束住。

模型的职责被压到最小：**挑出值得说的判断，并为每一句标注出处。**
它不负责搬运数字（数字由程序装配），不负责编造出处（清单外的 id 会被拒）。

校验失败时把错误回灌让它重写——和 M0 的 ReAct 循环同一个原则：
**能修的人是模型，就把错误交给模型**。
"""

from __future__ import annotations

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from tra.agent.evidence import EvidencePack, build_evidence
from tra.agent.llm import LLMCallable, extract_json
from tra.report.schema import Claim, Section, SectionKind

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
MAX_CLAIMS_PER_SECTION = 4
MAX_SECTIONS = 4

DRAFTABLE_SECTIONS: tuple[SectionKind, ...] = (
    SectionKind.SUMMARY,
    SectionKind.FINANCIALS,
    SectionKind.VALUATION,
    SectionKind.RISKS,
)

SYSTEM_PROMPT = """You are an equity research analyst writing a factual, cited report.

Hard rules:
- Every claim MUST cite at least one source id from the SOURCES list. Never invent an id.
- Every number you write MUST appear verbatim in the METRICS block. Do not round,
  restate or recall figures from memory.
- Do NOT compute differences, sums or ratios yourself. To describe a change, either give
  the two endpoint values ("fell from 74.6% to 60.5%") or describe it in words
  ("about fourteen points"). Never write the arithmetic result as a figure.
- Never give a rating, recommendation or price target.
- If the evidence does not support a claim, omit the claim. Gaps must be stated plainly.
- Cite at most 3 sources per claim. If a claim needs more, it is too broad — split it.
- Periods in METRICS may not be consecutive. Only use "consecutive", "sequentially" or
  "quarter over quarter" when the two period labels are genuinely adjacent (FY2026Q2
  follows FY2026Q1; FY2026Q1 does NOT follow FY2025Q3).
- Be specific. "Revenue grew" is useless; "Gross margin fell to 60.5% in FY2026Q1" is not.
- Do NOT walk through the table row by row. The reader already sees every number in the
  metrics table. Write only what the numbers MEAN: turning points, divergences between
  metrics, things that need explaining. One metric restated per claim is wasted output.
- At most 4 claims per section, 4 sections. Fewer, sharper claims beat exhaustive ones.

Return ONLY a JSON object of this exact shape:

{
  "sections": [
    {
      "kind": "summary" | "financials" | "valuation" | "risks",
      "title": "short human title",
      "claims": [
        {"text": "one factual sentence",
         "source_ids": ["<id from SOURCES>"],
         "confidence": "high" | "medium" | "low"}
      ]
    }
  ]
}"""


class DraftClaim(BaseModel):
    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1, max_length=3)
    """上限 3 条。给一句话挂 9 个出处等于没挂——读者无从核对哪个数字来自哪一份文件。"""
    confidence: Literal["high", "medium", "low"] = "medium"


class DraftSection(BaseModel):
    kind: SectionKind
    title: str = Field(min_length=1)
    claims: list[DraftClaim] = Field(default_factory=list)


class Draft(BaseModel):
    sections: list[DraftSection] = Field(default_factory=list)


def check_citations(draft: Draft, allowed: set[str]) -> list[str]:
    """列出所有编造的引用 id。返回空列表表示干净。"""
    bad: list[str] = []
    for section in draft.sections:
        for claim in section.claims:
            bad += [sid for sid in claim.source_ids if sid not in allowed]
    return sorted(set(bad))


_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> set[str]:
    """抓出文本里的数值，归一化成可比较的形式。

    只看三位及以上、或带小数点的数——"two quarters"、"10-Q" 这类小数字
    满地都是，检查它们只会制造噪音。真正危险的是 46,743 和 75.0% 这种
    看起来很具体、很可信的数字。
    """
    found: set[str] = set()
    for raw in _NUMBER_RE.findall(text):
        is_percent = raw.endswith("%")
        cleaned = raw.rstrip("%").replace(",", "").rstrip(".")
        if not cleaned:
            continue
        # 百分数一律核对，不看位数：模型最爱自己算的就是"占收入 33%"这种，
        if is_percent or "." in cleaned or len(cleaned.split(".")[0]) >= 3:
            found.add(cleaned.rstrip("0").rstrip(".") if "." in cleaned else cleaned)
    return found


def check_numbers(draft: Draft, evidence: str) -> list[str]:
    """找出报告里出现、但证据里没有的数字。

    这是"数字不经过模型"这条原则的**执行层**。前面的设计保证了报告表格里的
    数字来自程序，但模型仍然会在句子里复述数字——复述错了同样是幻觉，
    而且因为带着引用角标，比裸奔的错误更有欺骗性。

    真实案例：模型在报告里写"FY2027Q2 revenue appears as both $46,743 million
    and $96,221 million"，其中 46,743 根本不在证据包里——那是它自己的记忆。
    """
    allowed = numbers_in(evidence)
    unsupported: list[str] = []
    for section in draft.sections:
        for claim in section.claims:
            unsupported += [n for n in numbers_in(claim.text) if n not in allowed]
    return sorted(set(unsupported))


def to_sections(draft: Draft) -> list[Section]:
    """草稿 -> 报告契约里的 Section。丢弃空段落，超量的部分直接截断。

    **风格约束靠截断，正确性约束才打回。** 多写了两条判断不是错误，
    为它重跑一次模型要付出几分钟和一份钱；而编造引用是错误，必须打回。
    这两类约束的处理方式不该一样。
    """
    sections: list[Section] = []
    for drafted in draft.sections[:MAX_SECTIONS]:
        if drafted.kind not in DRAFTABLE_SECTIONS or not drafted.claims:
            continue
        sections.append(
            Section(
                kind=drafted.kind,
                title=drafted.title,
                claims=[
                    Claim(
                        text=claim.text,
                        source_ids=claim.source_ids,
                        confidence=claim.confidence,
                    )
                    for claim in drafted.claims[:MAX_CLAIMS_PER_SECTION]
                ],
            )
        )
    return sections


def draft_sections(
    pack: EvidencePack,
    question: str,
    llm: LLMCallable,
    *,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[list[Section], list[str]]:
    """起草段落。返回 (段落, 每次尝试的问题记录)。

    最多重试 max_attempts 次，每次把上一轮的具体错误回灌。
    全部失败时返回空段落——**宁可交白卷，也不交一份带假引用的报告**。
    """
    evidence = build_evidence(pack)
    problems: list[str] = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"RESEARCH QUESTION: {question}\n\n{evidence}"},
    ]

    for attempt in range(1, max_attempts + 1):
        log.info("起草报告，第 %d/%d 轮…", attempt, max_attempts)
        raw = llm(messages)

        try:
            draft = Draft.model_validate(extract_json(raw))
        except (ValueError, ValidationError) as exc:
            problem = f"输出不符合约定格式：{exc}"
            problems.append(problem)
            log.warning("第 %d 轮被打回：%s", attempt, problem)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"{problem}\n请只返回符合 schema 的 JSON。"},
            ]
            continue

        unsupported = check_numbers(draft, evidence)
        if unsupported:
            problem = f"用了证据里没有的数字：{unsupported}"
            problems.append(problem)
            log.warning("第 %d 轮被打回：%s", attempt, problem)
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"{problem}\n每个数字都必须逐字出现在 METRICS 里。"
                        f"记不准就别写数字，用文字描述方向和量级。"
                    ),
                },
            ]
            continue

        fabricated = check_citations(draft, pack.allowed_source_ids)
        if fabricated:
            problem = f"引用了不存在的 source id：{fabricated}"
            problems.append(problem)
            log.warning("第 %d 轮被打回：%s", attempt, problem)
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"{problem}\n只能引用 SOURCES 清单里的 id。"
                        f"无法归因的判断请整句删掉，不要换个 id 硬凑。"
                    ),
                },
            ]
            continue

        sections = to_sections(draft)
        if not sections:
            problem = "没有产出任何可用段落"
            problems.append(problem)
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"{problem}\n请至少给出 summary 段落。"},
            ]
            continue

        log.info("草稿通过校验：%d 个段落", len(sections))
        return sections, problems

    log.error("%d 轮都没通过校验，交白卷", max_attempts)
    return [], problems
