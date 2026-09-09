"""LLM 评判：确定性指标测不到的那部分。

**只用来评"读起来怎么样"，不用来判对错。** 事实性、可核查性、合规边界
全部在 metrics.py 里用纯函数判定——一个用模型判断模型的指标，
出问题时你分不清是谁错了。

评判维度是按**已经踩过的坑**选的，不是抄来的通用清单：

- internal_consistency  M3 并行之后出现的新失败模式：两个 researcher 互相矛盾
- specificity           M2 早期的毛病：逐行复述表格，零信息增量
- insight               判断有没有超出"数字是多少"
- honesty               缺口、局限有没有如实交代
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from tra.agent.llm import LLMCallable, extract_json
from tra.report.render import render_markdown
from tra.report.schema import Report

log = logging.getLogger(__name__)

Score = Literal[1, 2, 3, 4, 5]

JUDGE_PROMPT = """You are grading an automated equity research report. You are NOT
checking arithmetic or citations — those are verified separately by code.

Grade only these four dimensions, 1-5 each:

- internal_consistency: do any two claims contradict each other? A report that calls the
  same event "structural" in one place and "transitory" in another scores 1-2.
- specificity: does each claim say something a reader could not get by glancing at the
  table? Restating rows ("revenue was X in Q1, Y in Q2") scores 1-2.
- insight: does the report identify turning points, divergences between metrics, or
  anomalies worth explaining? Listing facts without interpretation scores 1-2.
- honesty: does it state plainly what the data cannot answer, instead of filling gaps
  with plausible-sounding claims? Confident claims beyond the evidence score 1.

Return ONLY JSON:
{"internal_consistency": 1-5, "specificity": 1-5, "insight": 1-5, "honesty": 1-5,
 "contradictions": ["quote the conflicting pair, if any"],
 "comment": "one sentence, the single biggest weakness"}"""


class JudgeScores(BaseModel):
    internal_consistency: Score = 3
    specificity: Score = 3
    insight: Score = 3
    honesty: Score = 3
    contradictions: list[str] = Field(default_factory=list)
    comment: str = ""

    @property
    def mean(self) -> float:
        return (self.internal_consistency + self.specificity + self.insight + self.honesty) / 4


def judge_report(report: Report, llm: LLMCallable) -> JudgeScores | None:
    """给一份报告打分。评判失败时返回 None——**绝不用默认分蒙混过去**。

    一个"评判挂了就给 3 分"的评测系统，会让你在趋势图上看到一条平稳的假线。
    宁可这一格空着。
    """
    messages = [
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": render_markdown(report)},
    ]
    try:
        return JudgeScores.model_validate(extract_json(llm(messages)))
    except (ValueError, ValidationError) as exc:
        log.warning("评判失败：%s", exc)
        return None
