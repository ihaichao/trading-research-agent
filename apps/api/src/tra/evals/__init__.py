"""评测：把"我的改动让报告变好了"从口头声明变成可验证事实。"""

from tra.evals.cases import EvalCase, load_cases
from tra.evals.judge import JudgeScores, judge_report
from tra.evals.metrics import ReportMetrics, evaluate_report
from tra.evals.runner import run_case, run_dataset, summarize, write_results

__all__ = [
    "EvalCase",
    "JudgeScores",
    "ReportMetrics",
    "evaluate_report",
    "judge_report",
    "load_cases",
    "run_case",
    "run_dataset",
    "summarize",
    "write_results",
]
