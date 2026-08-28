"""M0-2 第二步：ReAct 主循环的验收测试。

这些测试**不需要网络、不需要 API key、不花钱**——因为 `run_react` 把 LLM
作为参数接收，测试塞一个按剧本返回固定文本的假模型进去。

这就是依赖注入在 agent 开发里的价值：模型输出不确定、慢、要花钱，
把它挡在参数边界外，循环本身的逻辑就变成了普通的、可确定性测试的代码。
LangGraph / LangChain 也是这么设计的，你之后会反复见到。
"""

from __future__ import annotations

from typing import Any

import pytest
from examples import react_from_scratch as react
from examples.react_from_scratch import MAX_STEPS, run_react


class FakeLLM:
    """按剧本依次吐出预设文本，并录下每次被调用时收到的完整对话历史。

    录下历史这件事很关键：循环有没有把模型自己的输出放回去、有没有把工具结果
    回灌，光看返回值是测不出来的，必须检查它喂给模型的东西。
    """

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        # 深拷贝一份，否则后续对 messages 的追加会改到已录下的快照
        self.calls.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError(
                f"run_react 调用 LLM 的次数超出剧本（已调用 {len(self.calls)} 次）"
            )
        return self.responses.pop(0)

    def transcript(self, call_index: int) -> str:
        """把第 N 次调用时的全部消息拼成一段文本，方便做包含断言。"""
        return "\n".join(str(m.get("content", "")) for m in self.calls[call_index])


@pytest.fixture
def echo_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    """临时往 TOOLS 里塞一个可预测的工具，测试结束自动还原。"""
    monkeypatch.setitem(react.TOOLS, "echo", lambda s: f"ECHO:{s}")


def test_returns_final_answer_without_calling_tools() -> None:
    """模型第一句就给最终答案时，循环应该立刻收工。"""
    llm = FakeLLM(["Thought: 这个我知道。\nFinal Answer: 42"])
    assert run_react("生命的意义是什么？", llm=llm, verbose=False) == "42"
    assert len(llm.calls) == 1, "不该有多余的模型调用"


def test_first_call_starts_with_system_then_user() -> None:
    """第一次调用必须带上系统提示词和用户问题，否则模型不知道输出格式。"""
    llm = FakeLLM(["Final Answer: ok"])
    run_react("NVDA 多少钱？", llm=llm, verbose=False)

    first = llm.calls[0]
    assert first[0]["role"] == "system"
    assert "Available tools" in first[0]["content"]
    assert any(m["role"] == "user" and "NVDA 多少钱？" in m["content"] for m in first)


@pytest.mark.usefixtures("echo_tool")
def test_runs_the_tool_and_feeds_the_observation_back() -> None:
    """工具执行结果必须回到模型眼前，否则它下一步还是瞎猜。"""
    llm = FakeLLM(
        [
            "Thought: 我需要查一下。\nAction: echo\nAction Input: hi",
            "Thought: 拿到了。\nFinal Answer: 结果是 ECHO:hi",
        ]
    )
    assert run_react("q", llm=llm, verbose=False) == "结果是 ECHO:hi"
    assert len(llm.calls) == 2

    assert "ECHO:hi" in llm.transcript(1), "工具结果没有回灌给模型"


@pytest.mark.usefixtures("echo_tool")
def test_model_output_is_added_to_history() -> None:
    """模型自己说过的话也必须放回历史——LLM 是无状态的，不放它就失忆了。

    失忆的表现：模型看不到自己上一步决定调什么工具，于是重复调用同一个工具，
    循环空转直到步数用尽。
    """
    llm = FakeLLM(
        [
            "Thought: 查一下。\nAction: echo\nAction Input: hi",
            "Final Answer: done",
        ]
    )
    run_react("q", llm=llm, verbose=False)

    second = llm.transcript(1)
    assert "Action: echo" in second, "模型自己的输出没有放回 messages（会失忆）"
    assert any(m["role"] == "assistant" for m in llm.calls[1]), (
        "模型的输出应该以 assistant 角色追加"
    )


def test_unknown_tool_is_fed_back_instead_of_crashing() -> None:
    """模型编了一个不存在的工具名，程序不能崩，要把错误告诉它让它改。"""
    llm = FakeLLM(
        [
            "Thought: 试试这个。\nAction: no_such_tool\nAction Input: x",
            "Thought: 那个工具不存在，我换一个说法。\nFinal Answer: recovered",
        ]
    )
    assert run_react("q", llm=llm, verbose=False) == "recovered"
    assert "no_such_tool" in llm.transcript(1), "错误信息应该回灌给模型"


def test_tool_exception_is_fed_back_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """工具自己抛异常（网络挂了、接口变了）同样不能让整轮研究白跑。"""

    def boom(_: str) -> str:
        raise RuntimeError("upstream API is down")

    monkeypatch.setitem(react.TOOLS, "boom", boom)

    llm = FakeLLM(
        [
            "Thought: 调一下。\nAction: boom\nAction Input: x",
            "Thought: 工具挂了，我如实说明。\nFinal Answer: recovered",
        ]
    )
    assert run_react("q", llm=llm, verbose=False) == "recovered"
    assert "upstream API is down" in llm.transcript(1), "异常信息应该回灌给模型"


@pytest.mark.usefixtures("echo_tool")
def test_gives_up_after_max_steps() -> None:
    """模型永远不给最终答案时，循环必须自己停下来，而且要诚实地说没结论。"""
    llm = FakeLLM(["Thought: 再查一次。\nAction: echo\nAction Input: hi"] * (MAX_STEPS + 3))

    result = run_react("q", llm=llm, verbose=False)

    assert isinstance(result, str)
    assert result.strip(), "放弃时也要返回一句人话，不能返回空字符串"
    assert len(llm.calls) <= MAX_STEPS, f"最多调用 {MAX_STEPS} 次，实际调用了 {len(llm.calls)} 次"
