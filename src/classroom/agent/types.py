"""Shared teacher/student agent records."""

from __future__ import annotations

from dataclasses import dataclass, field

from classroom.benchmark.humaneval import EvaluationResult
from classroom.harness.state import HarnessPatch


@dataclass(frozen=True)
class StudentFeature:
    name: str
    style: str
    bias: str


@dataclass
class StudentAttempt:
    student_id: str
    task_id: str
    completion: str
    rationale: str
    evaluation: EvaluationResult
    harness_snapshot: dict

    def to_dict(self) -> dict:
        return {
            "student_id": self.student_id,
            "task_id": self.task_id,
            "completion": self.completion,
            "rationale": self.rationale,
            "evaluation": {
                "passed": self.evaluation.passed,
                "result": self.evaluation.result,
                "score": self.evaluation.score,
            },
            "harness_snapshot": self.harness_snapshot,
        }


@dataclass
class TeachingAdvice:
    teacher_id: str
    student_id: str
    task_id: str
    score: float
    diagnosis: str
    patches: list[HarnessPatch] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "teacher_id": self.teacher_id,
            "student_id": self.student_id,
            "task_id": self.task_id,
            "score": self.score,
            "diagnosis": self.diagnosis,
            "patches": [patch.to_dict() for patch in self.patches],
        }
