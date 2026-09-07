"""M2 的验收：ticker -> 一份带引用的研究报告。

    uv run python examples/research_demo.py NVDA
    uv run python examples/research_demo.py NVDA "数据中心业务在放缓吗？"

产出两个文件到 .cache/reports/：
    NVDA.md      给人看的
    NVDA.json    给前端看的（符合 packages/contracts 里的 schema）
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from tra.agent import openai_llm, research
from tra.agent.pipeline import DEFAULT_QUESTION
from tra.config import get_settings
from tra.report import render_markdown

console = Console()


def setup_logging() -> None:
    """把库里的 log 打到终端。

    一个跑几分钟的 CLI **必须有进度输出**——否则用户无法区分"在跑"和"挂了"。
    这不是锦上添花：静默的长任务会被人 Ctrl-C 掉，然后当成 bug 报上来。

    用标准库 logging 而不是到处 print：库代码不该假设自己跑在终端里，
    它只负责发出事件，由入口决定怎么显示。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "NVDA").upper()
    question = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUESTION

    console.rule(f"[bold]{ticker}")
    console.print(f"[dim]{escape(question)}[/dim]\n")

    # json_mode=True 会让支持的端点强制返回 JSON；不支持的端点会忽略它，
    # 所以 synthesize 那边仍然自己校验，不依赖这个开关。
    result = research(ticker, question=question, llm=openai_llm(json_mode=True))

    if not result.ok or result.data is None:
        console.print("[red]研究失败[/red]")
        console.print(escape(result.for_model()))
        raise SystemExit(1)

    report = result.data
    out = Path(get_settings().cache_dir) / "reports"
    out.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(report)
    (out / f"{ticker}.md").write_text(markdown, encoding="utf-8")
    (out / f"{ticker}.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    console.print(escape(markdown))
    console.rule()
    console.print(
        f"{report.claim_count} 条判断 · {len(report.sources)} 个出处 · "
        f"{report.meta.duration_seconds:.0f}s"
    )
    console.print(f"已写出 {out / f'{ticker}.md'} 和 {out / f'{ticker}.json'}")
    if result.hint:
        console.print(f"\n[yellow]提示[/yellow] {escape(result.hint)}")


if __name__ == "__main__":
    main()
