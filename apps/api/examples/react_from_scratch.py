"""M0-2 练习：不依赖任何 agent 框架，手写一个最小 ReAct 循环。

目标：跑通 `uv run python examples/react_from_scratch.py "英伟达现在多少钱？"`

为什么要先写这个：LangGraph / create_agent 底下就是这个循环。自己写一遍，
面试时被问到"agent 到底是怎么工作的"，你答的是机制而不是 API 名字。

已经给你的：工具、提示词、LLM 调用、主函数。
需要你实现的：`parse_action()` 和 `run_react()` —— 也就是循环本身。

写完后把 tests/test_react_from_scratch.py 里的 `pytest.mark.skip` 删掉，
`make test` 应该全绿。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import yfinance as yf
from openai import OpenAI
from rich.console import Console

from tra.config import get_settings

console = Console()

MAX_STEPS = 6


# --------------------------------------------------------------------------
# 1. 工具层：普通 Python 函数，返回字符串。这一层完全不知道 LLM 的存在。
# --------------------------------------------------------------------------


def get_quote(ticker: str) -> str:
    """返回某个美股代码的最新价格。"""
    info = yf.Ticker(ticker.strip().upper()).fast_info
    price = info.get("last_price")
    currency = info.get("currency", "USD")
    if price is None:
        return f"ERROR: no price found for {ticker!r}"
    return f"{ticker.upper()} last price: {price:.2f} {currency}"


TOOLS: dict[str, Callable[[str], str]] = {
    "get_quote": get_quote,
}

TOOL_DOCS = "\n".join(f"- {name}(input: str): {fn.__doc__}" for name, fn in TOOLS.items())


# --------------------------------------------------------------------------
# 2. 提示词：ReAct 的全部魔法就在这段格式约定里。
# --------------------------------------------------------------------------

SYSTEM_PROMPT = f"""You are a research assistant. Answer the user's question by
reasoning step by step and calling tools when you need real data.

Available tools:
{TOOL_DOCS}

Respond in EXACTLY one of these two formats, and nothing else:

Thought: <your reasoning>
Action: <tool_name>
Action Input: <the argument, plain text>

...or, when you have enough information:

Thought: <your reasoning>
Final Answer: <the answer for the user>

Never invent numbers. If a tool returns an error, say so plainly.
"""


@dataclass
class Action:
    """模型想调用的一个工具。"""

    tool: str
    tool_input: str


# --------------------------------------------------------------------------
# 3. TODO(你来写)：解析模型输出
# --------------------------------------------------------------------------


def parse_action(text: str) -> Action | str:
    """解析模型的一次输出。

    返回 `Action` 表示还要继续调工具；返回 `str` 表示这是最终答案。

    要求：
      - 认出 "Final Answer:" 开头的最终答案，返回其后的文本（去掉首尾空白）
      - 认出 "Action:" / "Action Input:" 两行，返回对应的 Action
      - 两者都没有时，抛 ValueError（把原文带进异常信息，方便调试）
      - 大小写、多余空行都要能容错

    提示：re.search 配合 re.MULTILINE / re.DOTALL；先判 Final Answer 再判 Action。
    """
    raise NotImplementedError("M0-2: 实现我")


# --------------------------------------------------------------------------
# 4. TODO(你来写)：ReAct 主循环
# --------------------------------------------------------------------------


def run_react(question: str, *, verbose: bool = True) -> str:
    """跑一轮 ReAct，返回最终答案。

    骨架：
      1. messages = [system, user]
      2. 循环最多 MAX_STEPS 次：
         a. 调 `call_llm(messages)` 拿到一段文本
         b. 把这段文本作为 assistant 消息追加进 messages   ← 忘了这步，模型会失忆
         c. parse_action()
         d. 是 str  → 直接返回
            是 Action → 查 TOOLS 执行；工具不存在或抛异常时，
                        把错误信息当作 Observation 回灌，让模型自己纠错，
                        而不是让程序崩掉                    ← 这是 agent 健壮性的核心
         e. 把 "Observation: <结果>" 作为 user 消息追加
      3. 循环用尽仍没有 Final Answer → 返回一句诚实的"我没能在限定步数内得出结论"

    verbose=True 时用 console.print 打印每一步，你要能亲眼看到它在想什么。
    """
    raise NotImplementedError("M0-2: 实现我")


# --------------------------------------------------------------------------
# 5. 已经给你的：LLM 调用
# --------------------------------------------------------------------------


def call_llm(messages: list[dict[str, Any]]) -> str:
    """调用任意 OpenAI 兼容端点，返回纯文本。"""
    settings = get_settings()
    settings.require_llm()
    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,  # type: ignore[arg-type]
        temperature=settings.llm_temperature,
        stop=["Observation:"],  # 别让模型自己把观察结果编出来
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    question = " ".join(sys.argv[1:]) or "英伟达现在多少钱？"
    console.rule(f"[bold]{question}")
    answer = run_react(question)
    console.rule("[bold green]Final Answer")
    console.print(answer)


if __name__ == "__main__":
    main()
