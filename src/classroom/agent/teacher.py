"""Teacher agent that diagnoses attempts and proposes harness patches."""

from __future__ import annotations

import json

from classroom.agent.types import StudentAttempt, TeachingAdvice
from classroom.benchmark.humaneval import EvaluationResult
from classroom.harness.state import HarnessPatch, HarnessState
from classroom.model.client import ChatModel


class TeacherAgent:
    def __init__(self, teacher_id: str, model: ChatModel | None, harness: HarnessState) -> None:
        self.teacher_id = teacher_id
        self.model = model
        self.harness = harness

    def teach(self, attempt: StudentAttempt, peer_summary: str = "") -> TeachingAdvice:
        if self.model is not None:
            generated = self._teach_with_llm(attempt, peer_summary)
            if generated is not None:
                return generated
        return self._teach_with_rules(attempt, peer_summary)

    def observe_outcome(self, advice: TeachingAdvice, before: EvaluationResult, after: EvaluationResult) -> None:
        delta = after.score - before.score
        content = (
            f"Advice to {advice.student_id} on {advice.task_id} changed score by {delta:+.1f}. "
            f"Diagnosis was: {advice.diagnosis}"
        )
        self.harness.apply_patch(
            HarnessPatch(
                target="memory",
                content=content,
                reason="teacher self-evolution from advice outcome",
                source=self.teacher_id,
                name="teaching_outcome",
            )
        )

    def _teach_with_llm(self, attempt: StudentAttempt, peer_summary: str) -> TeachingAdvice | None:
        messages = [
            {"role": "system", "content": self.harness.render_system_prompt()},
            {
                "role": "user",
                "content": (
                    "Create teaching advice as JSON with fields diagnosis and patches. "
                    "Each patch must have target, content, reason, source, optional name.\n"
                    f"Student attempt:\n{json.dumps(attempt.to_dict(), ensure_ascii=False)}\n"
                    f"Peer summary:\n{peer_summary}"
                ),
            },
        ]
        try:
            data = json.loads(self.model.complete(messages))
            patches = [
                HarnessPatch(
                    target=item["target"],
                    content=item["content"],
                    reason=item.get("reason", "teacher patch"),
                    source=self.teacher_id,
                    name=item.get("name", ""),
                )
                for item in data.get("patches", [])
            ]
            return TeachingAdvice(
                teacher_id=self.teacher_id,
                student_id=attempt.student_id,
                task_id=attempt.task_id,
                score=attempt.evaluation.score,
                diagnosis=str(data.get("diagnosis", "")),
                patches=patches,
            )
        except (KeyError, TypeError, json.JSONDecodeError, ValueError):
            return None

    def _teach_with_rules(self, attempt: StudentAttempt, peer_summary: str) -> TeachingAdvice:
        if attempt.evaluation.passed:
            diagnosis = "Solution passed; preserve the current strategy and distill it into reusable memory."
            patches = [
                HarnessPatch(
                    target="memory",
                    content=f"Successful strategy on {attempt.task_id}: {attempt.rationale or 'write direct, testable code.'}",
                    reason="distill success into memory",
                    source=self.teacher_id,
                    name="success_memory",
                )
            ]
        else:
            diagnosis = f"Hidden tests failed with result `{attempt.evaluation.result}`. Improve edge-case reasoning and avoid overfitting examples."
            patches = [
                HarnessPatch(
                    target="skill",
                    content=(
                        "Before finalizing code, infer edge cases from the docstring, including empty inputs, "
                        "single-element inputs, duplicate values, and boundary numeric values."
                    ),
                    reason="failed HumanEval correctness check",
                    source=self.teacher_id,
                    name="edge_case_checklist",
                ),
                HarnessPatch(
                    target="tool_policy",
                    content="Use syntax checking and mentally simulate at least two nontrivial edge cases before returning completion.",
                    reason="student needs stronger verification loop",
                    source=self.teacher_id,
                    name="verification_policy",
                ),
            ]
        if peer_summary:
            patches.append(
                HarnessPatch(
                    target="memory",
                    content=f"Peer classroom pattern: {peer_summary}",
                    reason="transfer peer experience into student harness",
                    source=self.teacher_id,
                    name="peer_transfer",
                )
            )
        return TeachingAdvice(
            teacher_id=self.teacher_id,
            student_id=attempt.student_id,
            task_id=attempt.task_id,
            score=attempt.evaluation.score,
            diagnosis=diagnosis,
            patches=patches,
        )
