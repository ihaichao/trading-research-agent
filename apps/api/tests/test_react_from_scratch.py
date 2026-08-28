"""M0-2 的验收测试。

实现 examples/react_from_scratch.py 里的两个 TODO 之后，把下面这行删掉：

    pytestmark = pytest.mark.skip(...)

然后 `make test` 应该全绿。这些测试不需要网络、不需要 API key。
"""

from __future__ import annotations

import pytest
from examples.react_from_scratch import Action, parse_action


def test_parses_final_answer() -> None:
    text = "Thought: 我已经拿到价格了。\nFinal Answer: NVDA 现在是 123.45 美元。"
    assert parse_action(text) == "NVDA 现在是 123.45 美元。"


def test_parses_action() -> None:
    text = "Thought: 需要查一下行情。\nAction: get_quote\nAction Input: NVDA"
    result = parse_action(text)
    assert isinstance(result, Action)
    assert result.tool == "get_quote"
    assert result.tool_input == "NVDA"


def test_tolerates_extra_whitespace() -> None:
    text = "Thought: ok\n\nAction:   get_quote  \n\nAction Input:   NVDA  \n"
    result = parse_action(text)
    assert isinstance(result, Action)
    assert result.tool == "get_quote"
    assert result.tool_input == "NVDA"


def test_final_answer_wins_over_action() -> None:
    text = "Thought: done\nFinal Answer: 已经够了\nAction: get_quote"
    assert parse_action(text) == "已经够了"


def test_raises_on_garbage() -> None:
    with pytest.raises(ValueError):
        parse_action("我今天不太想按格式说话。")


# ---------------------------------------------------------------------------
# 下面四条是对抗性用例：模型不守规矩时会发生什么。
# 上面那五条只覆盖了 happy path，五条全绿的实现依然可能在真实对话里出错。
# ---------------------------------------------------------------------------


def test_final_answer_keeps_every_line() -> None:
    """最终答案可能有好几段，一行都不能丢。

    截断的答案比报错更糟：用户看不出少了东西。
    """
    text = (
        "Thought: done\n"
        "Final Answer: NVDA 收于 123.45 美元。\n"
        "这个价格来自 Yahoo Finance，可能有 15 分钟延迟。"
    )
    assert parse_action(text) == (
        "NVDA 收于 123.45 美元。\n这个价格来自 Yahoo Finance，可能有 15 分钟延迟。"
    )


def test_ignores_final_answer_mentioned_mid_sentence() -> None:
    """模型在 Thought 里谈论"最终答案"这件事，不等于它给出了最终答案。

    这是最危险的一类误判：循环会提前终止，把一句半截话当成研究结论返回，
    而且不报任何错。
    """
    text = (
        "Thought: 我还不能给出 Final Answer: 因为我还没查行情。\n"
        "Action: get_quote\n"
        "Action Input: NVDA"
    )
    result = parse_action(text)
    assert isinstance(result, Action), f"应该继续调工具，却当成最终答案返回了：{result!r}"
    assert result.tool == "get_quote"
    assert result.tool_input == "NVDA"


def test_ignores_action_mentioned_mid_sentence() -> None:
    """Thought 里提到工具名，不等于那一行是指令行。"""
    text = "Thought: 我应该用 Action: get_quote 这个工具。\nAction: get_quote\nAction Input: NVDA"
    result = parse_action(text)
    assert isinstance(result, Action)
    assert result.tool == "get_quote", f"工具名被污染了：{result.tool!r}"
    assert result.tool_input == "NVDA"


def test_action_input_stops_at_end_of_line() -> None:
    """Action Input 只取到行尾，后面的自言自语不属于参数。

    否则 get_quote 收到的 ticker 会是 "NVDA\\nThought: 等结果"。
    """
    text = "Thought: 查一下\nAction: get_quote\nAction Input: NVDA\nThought: 等结果"
    result = parse_action(text)
    assert isinstance(result, Action)
    assert result.tool_input == "NVDA", f"参数被污染了：{result.tool_input!r}"
