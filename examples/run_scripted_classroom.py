"""Offline smoke run for the teacher-student classroom.

This uses ScriptedChatModel so the orchestration can be inspected without an API
key. Real experiments should use scripts/run_teacher_student_classroom.py.
"""

from __future__ import annotations

from classroom.agent import StudentAgent, StudentFeature, TeacherAgent
from classroom.benchmark import HumanEvalBenchmark
from classroom.harness import HarnessState
from classroom.interaction import TeacherStudentClassroom
from classroom.model import ScriptedChatModel


def main() -> None:
    bad = '{"completion": "    return False\\n", "rationale": "baseline guess"}'
    good = (
        '{"completion": "    for i in range(len(numbers)):\\n'
        '        for j in range(i + 1, len(numbers)):\\n'
        '            if abs(numbers[i] - numbers[j]) < threshold:\\n'
        '                return True\\n'
        '    return False\\n", "rationale": "check all pairs"}'
    )
    student = StudentAgent(
        student_id="student_minimalist",
        feature=StudentFeature(name="minimalist", style="short direct code", bias="misses edge cases"),
        model=ScriptedChatModel([bad, good]),
        harness=HarnessState(
            identity="student_minimalist",
            base_prompt="Solve HumanEval tasks.",
            style="short direct code",
        ),
    )
    teacher = TeacherAgent(
        teacher_id="teacher",
        model=None,
        harness=HarnessState(
            identity="teacher",
            base_prompt="Teach coding students with structured harness patches.",
            style="diagnostic",
        ),
    )
    classroom = TeacherStudentClassroom(
        benchmark=HumanEvalBenchmark(task_ids=["HumanEval/0"]),
        teacher=teacher,
        students=[student],
        output_path="runs/classroom/scripted_rounds.jsonl",
        log_path="runs/classroom/scripted_transcript.md",
    )
    report = classroom.run(generations=1, tasks_per_generation=1)
    print(f"before={report.average_before:.1f} after={report.average_after:.1f} gain={report.gain:+.1f}")


if __name__ == "__main__":
    main()
