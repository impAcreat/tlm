"""Teacher-student classroom orchestration.

References:
- Multi-agent teacher/student feedback loops.
- AgentGym-style benchmark harness and trajectory logging.
- LEAP-style learning from privileged teacher feedback, but applied to harness
  patches rather than only model-weight updates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from classroom.agent.student import StudentAgent
from classroom.agent.teacher import TeacherAgent
from classroom.agent.types import StudentAttempt
from classroom.benchmark.humaneval import CodeTask, EvaluationResult, HumanEvalBenchmark
from classroom.interaction.peer import PeerExperienceBank


@dataclass
class ClassroomRound:
    generation: int
    task_id: str
    student_id: str
    before: EvaluationResult
    after: EvaluationResult
    advice: dict
    before_completion: str = ""
    after_completion: str = ""

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "task_id": self.task_id,
            "student_id": self.student_id,
            "before": {
                "passed": self.before.passed,
                "result": self.before.result,
                "score": self.before.score,
            },
            "after": {
                "passed": self.after.passed,
                "result": self.after.result,
                "score": self.after.score,
            },
            "advice": self.advice,
            "before_completion": self.before_completion,
            "after_completion": self.after_completion,
        }


@dataclass
class ClassroomReport:
    rounds: list[ClassroomRound] = field(default_factory=list)

    @property
    def average_before(self) -> float:
        return _mean([item.before.score for item in self.rounds])

    @property
    def average_after(self) -> float:
        return _mean([item.after.score for item in self.rounds])

    @property
    def gain(self) -> float:
        return self.average_after - self.average_before


class TeacherStudentClassroom:
    def __init__(
        self,
        benchmark: HumanEvalBenchmark,
        teacher: TeacherAgent,
        students: list[StudentAgent],
        output_path: str | Path,
        log_path: str | Path | None = None,
    ) -> None:
        self.benchmark = benchmark
        self.teacher = teacher
        self.students = students
        self.output_path = Path(output_path)
        self.log_path = Path(log_path) if log_path else None
        self.peer_bank = PeerExperienceBank()
        self.use_pocketflow = True

    def run(self, generations: int = 1, tasks_per_generation: int = 2) -> ClassroomReport:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("# Classroom Transcript\n\n", encoding="utf-8")
        report = ClassroomReport()
        with self.output_path.open("w", encoding="utf-8") as file:
            for generation in range(1, generations + 1):
                tasks = self._select_tasks(generation, tasks_per_generation)
                for task in tasks:
                    for student in self.students:
                        round_result = self._teach_one(generation, task, student)
                        report.rounds.append(round_result)
                        file.write(json.dumps(round_result.to_dict(), ensure_ascii=False) + "\n")
                        self._append_log(round_result)
        return report

    def _teach_one(self, generation: int, task: CodeTask, student: StudentAgent) -> ClassroomRound:
        if self.use_pocketflow:
            from classroom.interaction.workflow import run_round_with_pocketflow

            return run_round_with_pocketflow(
                {
                    "generation": generation,
                    "task": task,
                    "student": student,
                    "teacher": self.teacher,
                    "benchmark": self.benchmark,
                    "peer_bank": self.peer_bank,
                }
            )

        completion, rationale = student.solve(task)
        before = self.benchmark.evaluate(task, completion)
        attempt = StudentAttempt(
            student_id=student.student_id,
            task_id=task.task_id,
            completion=completion,
            rationale=rationale,
            evaluation=before,
            harness_snapshot=student.harness.to_dict(),
        )
        self.peer_bank.add_attempt(attempt)

        peer_summary = self.peer_bank.summarize_for_task(task.task_id)
        advice = self.teacher.teach(attempt, peer_summary=peer_summary)
        self.peer_bank.add_advice(advice)
        student.apply_advice(advice)

        revised_completion, _ = student.solve(task)
        after = self.benchmark.evaluate(task, revised_completion)
        self.teacher.observe_outcome(advice, before, after)
        return ClassroomRound(
            generation=generation,
            task_id=task.task_id,
            student_id=student.student_id,
            before=before,
            after=after,
            advice=advice.to_dict(),
            before_completion=completion,
            after_completion=revised_completion,
        )

    def _select_tasks(self, generation: int, limit: int) -> list[CodeTask]:
        start = ((generation - 1) * limit) % len(self.benchmark.tasks)
        rotated = self.benchmark.tasks[start:] + self.benchmark.tasks[:start]
        return rotated[:limit]

    def _append_log(self, round_result: ClassroomRound) -> None:
        if self.log_path is None:
            return
        advice = round_result.advice
        text = (
            f"## Generation {round_result.generation} | {round_result.task_id} | {round_result.student_id}\n\n"
            f"- Before: {_format_score(round_result.before)}\n"
            f"- Teacher score: {advice.get('score', 0.0):.1f}\n"
            f"- Diagnosis: {advice.get('diagnosis', '')}\n"
            f"- Patches:\n{_format_patches(advice)}\n"
            f"- After: {_format_score(round_result.after)}\n\n"
        )
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(text)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _format_patches(advice: dict) -> str:
    patches = advice.get("patches", [])
    if not patches:
        return "- none"
    lines = []
    for patch in patches:
        target = patch.get("target", "unknown")
        name = patch.get("name", "")
        reason = patch.get("reason", "")
        content = patch.get("content", "")
        label = f"{target}:{name}" if name else target
        lines.append(f"- `{label}`: {content} ({reason})")
    return "\n".join(lines)


def _format_score(result: EvaluationResult) -> str:
    status = "PASS" if result.passed else "FAIL"
    return f"{status} score={result.score:.1f} result={result.result}"
