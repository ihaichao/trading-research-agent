"""评测用例。

三类用例，各自测不同的东西：

- **factual**  有唯一正确答案，报告里必须出现某个具体数字或事实
- **rubric**   开放问题，只能靠打分（确定性指标 + LLM 评判）
- **trap**     **陷阱题：正确答案是"这个数据回答不了"**

第三类最重要，也最容易被忽略。真实运行里模型写过
"营收集中在 Q1 和 Q2，存在季节性风险"——数据里根本没有完整财年，
这个结论无从谈起。它没编数字、没编引用，所有机械校验都放行了。

**衡量一个研究 agent，"该沉默时是否沉默"和"该说话时说得对不对"同样重要。**
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

CaseKind = Literal["factual", "rubric", "trap"]


class EvalCase(BaseModel):
    id: str
    kind: CaseKind
    ticker: str
    question: str

    must_contain: list[str] = Field(
        default_factory=list,
        description="factual 用例：报告里必须出现的字符串（数字按渲染后的形式写）",
    )
    must_not_contain: list[str] = Field(
        default_factory=list,
        description="trap 用例：报告里出现即算失败的字符串，比如凭空捏造的口径",
    )
    must_refuse_about: list[str] = Field(
        default_factory=list,
        description=(
            "trap 用例：报告必须明确说明这些东西无法从证据得出。只要提到它却不承认局限，就算失败。"
        ),
    )
    note: str = ""


REFUSAL_MARKERS = (
    "cannot be",
    "can not be",
    "not present",
    "not provided",
    "not available",
    "not in the",
    "no data",
    "unavailable",
    "cannot be assessed",
    "cannot be determined",
    "insufficient",
    "does not contain",
)
"""模型承认局限时的常见说法。

刻意做成关键词列表而不是 LLM 判断：**评测的地基必须比被测系统更可靠**。
一个用模型判断模型的指标，出问题时你分不清是谁错了。
"""


def load_cases(path: str | Path) -> list[EvalCase]:
    """从 jsonl 读用例。一行一个，方便 diff 和逐条追加。"""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        EvalCase.model_validate(json.loads(line))
        for line in lines
        if line.strip() and not line.lstrip().startswith("//")
    ]
