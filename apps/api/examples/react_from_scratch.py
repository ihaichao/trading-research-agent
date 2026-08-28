from __future__ import annotations

import re
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
    t = yf.Ticker(ticker.strip().upper())
    info = t.fast_info
    price = info.get("lastPrice") or getattr(info, "last_price", None)
    currency = info.get("currency", "USD") if hasattr(info, "get") else "USD"
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
# 3. 解析模型输出
# --------------------------------------------------------------------------


def parse_action(text: str) -> Action | str:
    """解析模型的一次输出。"""
    final_match = re.search(
        r"^\s*Final Answer:\s*(.*?)(?=\n\s*Action:|\Z)",
        text,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if final_match:
        return final_match.group(1).strip()

    action_match = re.search(
        r"^\s*Action:\s*([^\n]+)\s*\n+\s*Action Input:\s*([^\n]*)",
        text,
        re.IGNORECASE | re.MULTILINE,
    )

    if action_match:
        tool_name = action_match.group(1).strip()
        tool_input = action_match.group(2).strip()
        if tool_name:
            return Action(tool=tool_name, tool_input=tool_input)

    raise ValueError(f"Could not parse LLM output into Action or Final Answer:\n{text}")


# --------------------------------------------------------------------------
# 4. LLM 调用
# --------------------------------------------------------------------------

# 「能把一段对话历史变成一段文本」的任何东西。真实实现是下面的 call_llm，
# 测试里塞一个按剧本返回固定文本的假模型。
LLMCallable = Callable[[list[dict[str, Any]]], str]


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


# --------------------------------------------------------------------------
# 5. ReAct 主循环
# --------------------------------------------------------------------------


def run_react(
    question: str,
    *,
    llm: LLMCallable = call_llm,
    verbose: bool = True,
) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for step in range(MAX_STEPS):
        if verbose:
            console.rule(f"[bold yellow]Step {step + 1}/{MAX_STEPS}")

        # a. 调 llm 拿到一段文本
        llm_output = llm(messages)
        if verbose:
            console.print(f"[cyan]LLM Output:[/cyan]\n{llm_output}\n")

        # b. 把模型输出作为 assistant 消息追加（防止失忆）
        messages.append({"role": "assistant", "content": llm_output})

        # c. 解析动作
        try:
            action_or_answer = parse_action(llm_output)
        except ValueError as e:
            observation = f"ERROR: Invalid format ({e})"
            messages.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        # d. 最终答案直接返回
        if isinstance(action_or_answer, str):
            return action_or_answer

        # e. 是 Action，执行工具并回灌 Observation
        action = action_or_answer
        tool_fn = TOOLS.get(action.tool)
        if tool_fn is None:
            observation = (
                f"ERROR: tool {action.tool!r} not found. Available tools: {list(TOOLS.keys())}"
            )
        else:
            try:
                observation = tool_fn(action.tool_input)
            except Exception as e:
                observation = f"ERROR executing tool {action.tool}: {e}"

        if verbose:
            console.print(f"[green]Observation:[/green] {observation}\n")

        messages.append({"role": "user", "content": f"Observation: {observation}"})

    return "Sorry, I couldn't get the answer in the given steps."


def main() -> None:
    question = " ".join(sys.argv[1:]) or "英伟达现在多少钱？"
    console.rule(f"[bold]{question}")
    answer = run_react(question)
    console.rule("[bold green]Final Answer")
    console.print(answer)


if __name__ == "__main__":
    main()
