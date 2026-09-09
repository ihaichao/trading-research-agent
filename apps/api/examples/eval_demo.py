"""M5 的验收：跑一遍评测集。

    uv run python examples/eval_demo.py                 # 只跑确定性指标（免费）
    uv run python examples/eval_demo.py --judge         # 加上 LLM 评判（花钱）
    uv run python examples/eval_demo.py --only nvda     # 只跑 id 含 nvda 的用例

结果写到 evals/results/<日期>.json 和 .md。
**改动前后各跑一次，把两份 md 放一起 diff** —— 这就是"我的改动让报告变好了"
从口头声明变成可验证事实的全部机制。
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape

from tra.agent.llm import openai_llm
from tra.evals import load_cases, run_dataset, write_results

console = Console()
DATASET = Path(__file__).resolve().parents[1] / "evals" / "dataset.jsonl"
RESULTS = Path(__file__).resolve().parents[1] / "evals" / "results"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[RichHandler(console=console, show_path=False)],
    )
    for noisy in ("httpx", "openai", "edgar"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    cases = load_cases(DATASET)
    if "--only" in sys.argv:
        needle = sys.argv[sys.argv.index("--only") + 1].lower()
        cases = [c for c in cases if needle in c.id.lower()]

    use_judge = "--judge" in sys.argv
    console.rule(f"[bold]评测 {len(cases)} 条用例{'（含 LLM 评判）' if use_judge else ''}")

    llm = openai_llm(json_mode=True)
    result = run_dataset(cases, llm, judge_llm=openai_llm(json_mode=True) if use_judge else None)

    json_path, md_path = write_results(result, RESULTS)
    console.print()
    console.print(escape(md_path.read_text(encoding="utf-8")))
    console.rule()
    console.print(f"已写出 {md_path} 和 {json_path}")


if __name__ == "__main__":
    main()
