"""模型调用层。

只做一件事：把"一段对话"变成"一段文本"。所有和具体厂商相关的东西
（base_url、模型名、鉴权）都锁在这里，上层拿到的是一个普通函数。

"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from tra.config import get_settings

Message = dict[str, str]
LLMCallable = Callable[[list[Message]], str]

log = logging.getLogger(__name__)


def openai_llm(
    *,
    temperature: float | None = None,
    json_mode: bool = False,
    stream: bool = True,
) -> LLMCallable:
    """构造一个调用 OpenAI 兼容端点的函数。

    兼容 OpenAI / DeepSeek / Moonshot / OpenRouter —— 换供应商只改 .env，
    不动代码。json_mode 在支持的端点上强制返回 JSON；不支持时会被忽略，
    所以**上层仍然必须自己校验**。

    默认开流式：它不会让生成变快，但让"还在生成"和"已经挂了"变得可区分。
    一个要等好几分钟的调用，如果全程没有任何反馈，用户一定会当成死机。
    """

    def call(messages: list[Message]) -> str:
        from openai import OpenAI

        settings = get_settings()
        settings.require_llm()
        # 超时必须显式设置：默认值很长，一次卡住的请求会让整个流程静默挂死，
        # 而调用方看到的只是"没有反应"。
        client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

        kwargs: dict[str, Any] = {
            "model": settings.llm_model,
            "messages": messages,
            "temperature": settings.llm_temperature if temperature is None else temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        log.info(
            "调用模型 %s（提示词约 %d 字符，超时 %.0fs%s）…",
            settings.llm_model,
            prompt_chars,
            settings.llm_timeout_seconds,
            "，流式" if stream else "",
        )
        started = time.monotonic()

        if stream:
            try:
                text = _stream(client, kwargs, started)
            except Exception as exc:
                # 有的端点不支持流式。降级重来一次，别因为一个可选特性整个失败。
                log.warning("流式失败（%s），改用非流式重试", type(exc).__name__)
                text = _once(client, kwargs)
        else:
            text = _once(client, kwargs)

        elapsed = time.monotonic() - started
        log.info(
            "模型返回 %d 字符，用时 %.1fs（%.0f 字符/秒）",
            len(text),
            elapsed,
            len(text) / elapsed if elapsed else 0,
        )
        return text

    return call


def _once(client: Any, kwargs: dict[str, Any]) -> str:
    response = client.chat.completions.create(**kwargs)
    return str(response.choices[0].message.content or "").strip()


def _stream(client: Any, kwargs: dict[str, Any], started: float) -> str:
    """边收边记进度。每 ~800 字符报一次，好让人知道它还活着。"""
    chunks: list[str] = []
    milestone = 0
    for event in client.chat.completions.create(**kwargs, stream=True):
        if not event.choices:
            continue
        piece = event.choices[0].delta.content
        if not piece:
            continue
        chunks.append(piece)
        total = sum(len(c) for c in chunks)
        if total - milestone >= 800:
            milestone = total
            log.info("  …已生成 %d 字符（%.0fs）", total, time.monotonic() - started)
    return "".join(chunks).strip()


def extract_json(text: str) -> Any:
    """从模型输出里抠出 JSON。

    即使开了 json_mode，模型也常常给你套上 ```json 围栏，或者在前后加一句
    "好的，这是结果："。与其在提示词里反复求它别这么干，不如在解析这一侧宽容一点——
    **对模型的输出要宽进严出**：解析宽容，校验严格。
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(
            "模型返回了空字符串。常见原因：提示词过长、输出被 max_tokens 截断，"
            "或供应商侧超时。先看证据包是不是太大。"
        )
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[: -len("```")]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        return json.loads(cleaned[start : end + 1])
    raise ValueError(f"模型没有返回可解析的 JSON：{text[:200]!r}")
