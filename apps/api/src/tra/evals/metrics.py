"""确定性指标：不调模型、不花钱、可以进 CI。

**评测系统分两层,这是刻意的:**

  确定性层（这个文件）  免费、秒级、结果稳定 → 每次提交都跑
  LLM 评判层（judge.py）花钱、分钟级、有方差 → 按需跑

**评测的地基必须比被测系统更可靠**——所以这里全是纯函数，
没有任何一处需要模型参与。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from tra.agent.synthesize import numbers_in
from tra.evals.cases import REFUSAL_MARKERS, EvalCase
from tra.report.schema import Report

MAX_ATOMIC_CHARS = 400
MAX_SOURCES_PER_CLAIM = 3
DUPLICATE_THRESHOLD = 0.6

FORBIDDEN_TERMS = (
    "buy rating",
    "sell rating",
    "price target",
    "we recommend",
    "strong buy",
    "outperform",
    "underperform",
    "overweight",
)

SEASONALITY_TERMS = ("seasonal", "seasonality", "cyclical pattern")
ADJACENCY_TERMS = (
    "consecutive quarter",
    "sequentially",
    "quarter over quarter",
    "quarter-over-quarter",
)


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9.%]+", text.lower()) if len(w) > 2}


def _similarity(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


@dataclass
class ReportMetrics:
    """一份报告的确定性打分。每个字段都能在几毫秒内算出来。"""

    case_id: str
    ok: bool

    claim_count: int = 0
    source_count: int = 0
    sections: list[str] = field(default_factory=list)

    # 可核查性
    ungrounded_numbers: list[str] = field(default_factory=list)
    """报告里出现、证据里没有的数字。**这个必须恒为空**——不为空说明防线漏了。"""
    over_cited_claims: int = 0
    max_claim_chars: int = 0
    non_atomic_claims: int = 0

    # 内容质量
    max_duplicate_similarity: float = 0.0
    duplicate_pairs: int = 0

    # 合规与诚实
    forbidden_terms: list[str] = field(default_factory=list)
    unsupported_seasonality: bool = False
    unsupported_adjacency: bool = False
    discloses_gaps: bool = False

    # 用例判定
    missing_required: list[str] = field(default_factory=list)
    present_forbidden: list[str] = field(default_factory=list)
    failed_refusals: list[str] = field(default_factory=list)

    # 运行数据
    duration_seconds: float = 0.0
    dropped_claims: int = 0
    drop_reasons: list[str] = field(default_factory=list)
    """丢弃的原因原文。

    **只记数量，等于把一个静默错误换成一个静默数字。** 三道陷阱题都显示
    `dropped_claims: 2`，但光看这个数字无法知道被丢的是不是那条"我答不了"——
    而这恰恰是唯一需要知道的事。
    """

    def to_row(self) -> dict:
        from dataclasses import asdict

        return asdict(self) | {"passed": self.passed}

    @property
    def passed(self) -> bool:
        """硬性通过条件。软性质量（洞察力、可读性）交给 LLM 评判，不在这里判定。"""
        return (
            self.ok
            and not self.ungrounded_numbers
            and not self.forbidden_terms
            and not self.missing_required
            and not self.present_forbidden
            and not self.failed_refusals
            and not self.unsupported_seasonality
        )


def claim_texts(report: Report) -> list[str]:
    return [claim.text for section in report.sections for claim in section.claims]


def evaluate_report(
    case: EvalCase,
    report: Report | None,
    *,
    evidence: str = "",
    has_period_gaps: bool = False,
    duration_seconds: float = 0.0,
    dropped_claims: int = 0,
    drop_reasons: list[str] | None = None,
    error: str = "",
) -> ReportMetrics:
    """把一份报告（或一次失败）打成一行指标。"""
    metrics = ReportMetrics(
        case_id=case.id,
        ok=report is not None,
        duration_seconds=duration_seconds,
        dropped_claims=dropped_claims,
        drop_reasons=list(drop_reasons or []),
    )

    if report is None:
        # 报告没出来时，trap 类用例仍然可能是"通过"的：
        # 不存在的股票代码就该干净地失败。
        if case.kind == "trap" and case.must_refuse_about:
            haystack = error.lower()
            metrics.failed_refusals = [
                topic for topic in case.must_refuse_about if topic.lower() not in haystack
            ]
            metrics.ok = not metrics.failed_refusals
        return metrics

    texts = claim_texts(report)
    whole = "\n".join(texts).lower()

    metrics.claim_count = len(texts)
    metrics.source_count = len(report.sources)
    metrics.sections = [s.kind for s in report.sections]

    # --- 可核查性 ---
    if evidence:
        allowed = numbers_in(evidence)
        found: set[str] = set()
        for text in texts:
            found |= numbers_in(text) - allowed
        metrics.ungrounded_numbers = sorted(found)

    for section in report.sections:
        for claim in section.claims:
            if len(claim.source_ids) > MAX_SOURCES_PER_CLAIM:
                metrics.over_cited_claims += 1
            if len(claim.text) > MAX_ATOMIC_CHARS:
                metrics.non_atomic_claims += 1
            metrics.max_claim_chars = max(metrics.max_claim_chars, len(claim.text))

    # --- 重复 ---
    for section in report.sections:
        section_texts = [c.text for c in section.claims]
        for i, left in enumerate(section_texts):
            for right in section_texts[i + 1 :]:
                score = _similarity(left, right)
                metrics.max_duplicate_similarity = max(metrics.max_duplicate_similarity, score)
                if score >= DUPLICATE_THRESHOLD:
                    metrics.duplicate_pairs += 1

    # --- 合规与诚实 ---
    metrics.forbidden_terms = [term for term in FORBIDDEN_TERMS if term in whole]
    if has_period_gaps:
        metrics.unsupported_seasonality = _asserts_any(texts, SEASONALITY_TERMS)
        metrics.unsupported_adjacency = _asserts_any(texts, ADJACENCY_TERMS)
        metrics.discloses_gaps = any(
            marker in whole for marker in ("missing", "not consecutive", "absent", "gap")
        )

    # --- 用例判定 ---
    metrics.missing_required = [
        needle for needle in case.must_contain if needle not in "\n".join(texts)
    ]
    metrics.present_forbidden = [
        needle for needle in case.must_not_contain if needle.lower() in whole
    ]
    metrics.failed_refusals = [
        topic for topic in case.must_refuse_about if not _refuses_about(texts, topic)
    ]
    return metrics


def _asserts_any(texts: list[str], terms: tuple[str, ...]) -> bool:
    """是否**断言**了某类结论（而不是承认它无法判断）。"""
    for text in texts:
        lowered = text.lower()
        if any(term in lowered for term in terms) and not any(
            marker in lowered for marker in REFUSAL_MARKERS
        ):
            return True
    return False


def _refuses_about(texts: list[str], topic: str) -> bool:
    """报告是否明确承认某个主题回答不了。

    要求"提到该主题的那句话里同时带有承认局限的措辞"——
    只要提到却不承认局限，就算失败。**沉默不算承认**：
    完全不提这个主题也不算通过，因为用户问的就是它。
    """
    # 一项里可以用 | 写同义词，命中任意一个即可
    needles = [part.strip().lower() for part in topic.split("|") if part.strip()]
    mentions = [t.lower() for t in texts if any(needle in t.lower() for needle in needles)]
    if not mentions:
        return False
    return any(any(marker in text for marker in REFUSAL_MARKERS) for text in mentions)
