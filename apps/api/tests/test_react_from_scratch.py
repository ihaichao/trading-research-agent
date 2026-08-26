"""M0-2 的验收测试。

实现 examples/react_from_scratch.py 里的两个 TODO 之后，把下面这行删掉：

    pytestmark = pytest.mark.skip(...)

然后 `make test` 应该全绿。这些测试不需要网络、不需要 API key。
"""

from __future__ import annotations

import pytest
from examples.react_from_scratch import Action, parse_action

pytestmark = pytest.mark.skip(reason="M0-2 练习尚未完成；实现后删除这一行")


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
