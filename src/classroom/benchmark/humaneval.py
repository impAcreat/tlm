"""HumanEval benchmark adapter.

Uses OpenAI's `human-eval` package rather than reimplementing the benchmark.
The adapter narrows the surface area we need: load tasks, present prompts, and
execute the official correctness check.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from human_eval.data import read_problems
from human_eval.execution import check_correctness


@dataclass(frozen=True)
class CodeTask:
    task_id: str
    prompt: str
    entry_point: str
    raw_problem: dict[str, Any]


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    passed: bool
    result: str
    completion: str

    @property
    def score(self) -> float:
        return 1.0 if self.passed else 0.0


class HumanEvalBenchmark:
    def __init__(self, task_ids: list[str] | None = None, timeout_s: float = 3.0) -> None:
        problems = read_problems()
        if task_ids is None:
            selected = list(problems)[:8]
        else:
            selected = task_ids
        self.tasks = [
            CodeTask(
                task_id=task_id,
                prompt=problems[task_id]["prompt"],
                entry_point=problems[task_id]["entry_point"],
                raw_problem=problems[task_id],
            )
            for task_id in selected
        ]
        self.timeout_s = timeout_s

    def evaluate(self, task: CodeTask, completion: str) -> EvaluationResult:
        result = check_correctness(task.raw_problem, completion, timeout=self.timeout_s)
        return EvaluationResult(
            task_id=task.task_id,
            passed=bool(result["passed"]),
            result=str(result["result"]),
            completion=completion,
        )
