"""跑评测集，产出一份可 diff 的结果。

结果同时写两份：
  results/<日期>.json
  results/<日期>.md

**每次改动前后各跑一次,把两份 md 放一起 diff**
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from tra.agent.evidence import missing_periods
from tra.agent.llm import LLMCallable
from tra.evals.cases import EvalCase
from tra.evals.judge import JudgeScores, judge_report
from tra.evals.metrics import ReportMetrics, evaluate_report
from tra.graph import build_report, compile_graph
from tra.report.schema import MetricSeries, Report

log = logging.getLogger(__name__)


def run_case(case: EvalCase, llm: LLMCallable) -> tuple[ReportMetrics, Report | None]:
    """跑一个用例，返回指标和报告。"""
    started = time.monotonic()
    app = compile_graph(llm)
    report: Report | None = None
    evidence = ""
    gaps = False
    problems: list[str] = []
    error = ""

    try:
        state = app.invoke(
            {"ticker": case.ticker, "question": case.question, "findings": [], "problems": []},
            {"configurable": {"thread_id": f"eval-{case.id}"}},
        )
        evidence = state.get("evidence", "")
        problems = list(state.get("problems") or [])
        series = [MetricSeries.model_validate(s) for s in state.get("series") or []]
        gaps = bool(series and missing_periods(series[0]))
        report = build_report(state)
        problems = list(state.get("problems") or [])  # build_report 会补记去重/封顶丢弃
        if report is None:
            error = "; ".join(problems) or "no findings produced"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.warning("[%s] 运行失败：%s", case.id, error)

    metrics = evaluate_report(
        case,
        report,
        evidence=evidence,
        has_period_gaps=gaps,
        duration_seconds=time.monotonic() - started,
        dropped_claims=len(problems),
        drop_reasons=problems,
        error=error,
    )
    log.info(
        "[%s] %s · %d 条判断 · %.0fs",
        case.id,
        "通过" if metrics.passed else "未通过",
        metrics.claim_count,
        metrics.duration_seconds,
    )
    return metrics, report


def run_dataset(
    cases: list[EvalCase],
    llm: LLMCallable,
    *,
    judge_llm: LLMCallable | None = None,
) -> dict:
    """跑整个数据集。judge_llm 为 None 时只跑确定性指标（免费、可进 CI）。"""
    rows: list[dict] = []
    for case in cases:
        metrics, report = run_case(case, llm)
        row = metrics.to_row() | {"kind": case.kind, "ticker": case.ticker}
        if judge_llm and report is not None:
            scores = judge_report(report, judge_llm)
            row["judge"] = scores.model_dump() if scores else None
        rows.append(row)

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": len(rows),
        "summary": summarize(rows),
        "rows": rows,
    }


def summarize(rows: list[dict]) -> dict:
    """汇总。分 kind 统计，因为三类用例的通过意义完全不同。"""
    by_kind: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = by_kind.setdefault(row["kind"], {"passed": 0, "total": 0})
        bucket["total"] += 1
        bucket["passed"] += int(row["passed"])

    judged = [r["judge"] for r in rows if r.get("judge")]
    summary: dict = {
        "pass_rate": sum(r["passed"] for r in rows) / len(rows) if rows else 0.0,
        "by_kind": by_kind,
        "ungrounded_number_cases": sum(1 for r in rows if r["ungrounded_numbers"]),
        "median_duration_seconds": _median([r["duration_seconds"] for r in rows]),
        "total_claims": sum(r["claim_count"] for r in rows),
    }
    if judged:
        for dim in ("internal_consistency", "specificity", "insight", "honesty"):
            summary[f"judge_{dim}"] = round(sum(j[dim] for j in judged) / len(judged), 2)
    return summary


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 1)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 1)


def write_results(result: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """写出 json + md。文件名带日期，天然形成时间序列。"""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d-%H%M")

    json_path = directory / f"{stamp}.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = directory / f"{stamp}.md"
    md_path.write_text(render_results(result), encoding="utf-8")
    return json_path, md_path


def render_results(result: dict) -> str:
    summary = result["summary"]
    lines = [
        f"# 评测结果 {result['generated_at'][:16]}",
        "",
        f"- 用例 {result['cases']} 条，通过率 **{summary['pass_rate']:.0%}**",
        f"- 中位耗时 {summary['median_duration_seconds']}s，累计 {summary['total_claims']} 条 claim",
        f"- 未落地的数字：{summary['ungrounded_number_cases']} 个用例（应为 0）",
        "",
    ]
    for kind, bucket in sorted(summary["by_kind"].items()):
        lines.append(f"- `{kind}` {bucket['passed']}/{bucket['total']}")

    if any(k.startswith("judge_") for k in summary):
        lines += ["", "## LLM 评判（1–5）", ""]
        for dim in ("internal_consistency", "specificity", "insight", "honesty"):
            key = f"judge_{dim}"
            if key in summary:
                lines.append(f"- {dim}: **{summary[key]}**")

    lines += [
        "",
        "## 逐条",
        "",
        "| 用例 | 类型 | 结果 | claim 数 | 耗时 | 问题 |",
        "|---|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        issues = []
        if row["ungrounded_numbers"]:
            issues.append(f"未落地数字 {row['ungrounded_numbers']}")
        if row["missing_required"]:
            issues.append(f"缺少 {row['missing_required']}")
        if row["failed_refusals"]:
            issues.append(f"未承认局限 {row['failed_refusals']}")
        if row["forbidden_terms"]:
            issues.append(f"违禁措辞 {row['forbidden_terms']}")
        if row["unsupported_seasonality"]:
            issues.append("无依据的季节性结论")
        if row["duplicate_pairs"]:
            issues.append(f"重复判断 {row['duplicate_pairs']} 对")
        lines.append(
            f"| {row['case_id']} | {row['kind']} | {'✅' if row['passed'] else '❌'} "
            f"| {row['claim_count']} | {row['duration_seconds']:.0f}s "
            f"| {'; '.join(issues) or '—'} |"
        )

    # **被丢掉的判断，往往才是失败的原因。** 通过率只告诉你哪一条挂了，
    # 这一节告诉你为什么。
    dropped = [r for r in result["rows"] if r.get("drop_reasons")]
    if dropped:
        lines += ["", "## 被丢弃的判断", ""]
        for row in dropped:
            lines.append(f"**{row['case_id']}**")
            lines += [f"- {reason}" for reason in row["drop_reasons"]]
            lines.append("")

    return "\n".join(lines) + "\n"


__all__ = ["JudgeScores", "run_case", "run_dataset", "summarize", "write_results"]
