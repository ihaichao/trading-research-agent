"""M3 的验收：用 LangGraph 跑一遍「计划 → 并行研究 → 装配」。

    uv run python examples/graph_demo.py NVDA
    uv run python examples/graph_demo.py NVDA --pause      # 规划后暂停，确认大纲再继续

和 M2 那条直线比，看三件事：
  1. 日志里多个 [research:xxx] 交错出现 —— 它们在并行
  2. 总耗时接近最慢的那个 researcher，而不是全部之和
  3. --pause 能在规划后停下、打印大纲、等你回车再继续
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from tra.agent.llm import openai_llm
from tra.agent.pipeline import DEFAULT_QUESTION
from tra.config import get_settings
from tra.graph import build_report, compile_graph
from tra.graph.state import SubQuestion
from tra.report import render_markdown

console = Console()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )
    for noisy in ("httpx", "openai", "edgar"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> None:
    setup_logging()
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pause = "--pause" in sys.argv

    ticker = (args[0] if args else "NVDA").upper()
    question = args[1] if len(args) > 1 else DEFAULT_QUESTION

    console.rule(f"[bold]{ticker}")
    console.print(f"[dim]{escape(question)}[/dim]\n")

    cache = Path(get_settings().cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC)

    # checkpointer 落盘：中途崩了、或者 --pause 停下，进度都还在
    with SqliteSaver.from_conn_string(str(cache / "graph.sqlite")) as checkpointer:
        app = compile_graph(
            openai_llm(json_mode=True), checkpointer=checkpointer, pause_after_plan=pause
        )
        config = {"configurable": {"thread_id": f"{ticker}-{started:%Y%m%d%H%M%S}"}}
        state = {"ticker": ticker, "question": question, "findings": [], "problems": []}

        result = app.invoke(state, config)

        if pause:
            console.rule("[yellow]规划完成，等待确认")
            for item in result.get("plan", []):
                sub = SubQuestion.model_validate(item)
                console.print(f"  [{sub.section}] {sub.key}: {escape(sub.question)}")
            console.input("\n回车继续，Ctrl-C 放弃 ")
            # 人思考的时间不该算进"研究耗时"——否则这个数字失去意义
            started = datetime.now(UTC)
            result = app.invoke(None, config)

    report = build_report(result, started=started)
    if report is None:
        console.print("[red]没有产出任何发现[/red]")
        for problem in result.get("problems", []):
            console.print(f"  {escape(problem)}")
        raise SystemExit(1)

    out = cache / "reports"
    out.mkdir(parents=True, exist_ok=True)
    markdown = render_markdown(report)
    (out / f"{ticker}-graph.md").write_text(markdown, encoding="utf-8")
    (out / f"{ticker}-graph.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    console.print(escape(markdown))
    console.rule()
    console.print(
        f"{report.claim_count} 条判断 · {len(report.sources)} 个出处 · "
        f"{len(result.get('plan', []))} 个并行 researcher · "
        f"{report.meta.duration_seconds:.0f}s"
    )
    dropped = result.get("problems") or []
    if dropped:
        console.print(f"\n[yellow]被丢弃的判断 {len(dropped)} 条[/yellow]")
        for problem in dropped[:5]:
            console.print(f"  {escape(problem)}")


if __name__ == "__main__":
    main()
