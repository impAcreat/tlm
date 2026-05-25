"""Peer experience sharing for heterogeneous student agents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from classroom.agent.types import StudentAttempt, TeachingAdvice


@dataclass
class PeerExperienceBank:
    attempts: list[StudentAttempt] = field(default_factory=list)
    advice: list[TeachingAdvice] = field(default_factory=list)

    def add_attempt(self, attempt: StudentAttempt) -> None:
        self.attempts.append(attempt)

    def add_advice(self, advice: TeachingAdvice) -> None:
        self.advice.append(advice)

    def summarize_for_task(self, task_id: str) -> str:
        relevant = [attempt for attempt in self.attempts if attempt.task_id == task_id]
        if not relevant:
            return ""
        failures = [attempt for attempt in relevant if not attempt.evaluation.passed]
        styles = Counter(attempt.student_id for attempt in failures)
        if not failures:
            return "Peers solved this task; preserve successful reasoning patterns."
        common_results = Counter(attempt.evaluation.result for attempt in failures).most_common(2)
        return f"{len(failures)} peer failures; failing students={dict(styles)}; common_results={common_results}"
