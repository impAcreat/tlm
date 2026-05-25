from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from classroom.agent import StudentAgent, StudentFeature, TeacherAgent, parse_student_response
from classroom.benchmark import HumanEvalBenchmark
from classroom.harness import HarnessState
from classroom.interaction import TeacherStudentClassroom
from classroom.model import ScriptedChatModel


class TeacherStudentClassroomTest(TestCase):
    def test_student_response_parser_accepts_json(self) -> None:
        completion, rationale = parse_student_response(
            '{"completion": "    return True\\n", "rationale": "constant baseline"}'
        )

        self.assertEqual(completion, "    return True\n")
        self.assertEqual(rationale, "constant baseline")

    def test_student_response_parser_extracts_wrapped_json(self) -> None:
        completion, rationale = parse_student_response(
            '<think>check edge cases</think>\n'
            '```json\n'
            '{"completion": "    return len(numbers) > 1\\n", "rationale": "minimal"}\n'
            '```'
        )

        self.assertEqual(completion, "    return len(numbers) > 1\n")
        self.assertEqual(rationale, "minimal")

    def test_classroom_teaches_student_harness(self) -> None:
        bad = '{"completion": "    return False\\n", "rationale": "too simple"}'
        good = (
            '{"completion": "    for i in range(len(numbers)):\\n'
            '        for j in range(i + 1, len(numbers)):\\n'
            '            if abs(numbers[i] - numbers[j]) < threshold:\\n'
            '                return True\\n'
            '    return False\\n", "rationale": "check all pairs"}'
        )
        student_model = ScriptedChatModel([bad, good])
        feature = StudentFeature(name="minimalist", style="short direct code", bias="misses edge cases")
        student = StudentAgent(
            student_id="student_minimalist",
            feature=feature,
            model=student_model,
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
                base_prompt="Teach coding students.",
                style="diagnostic",
            ),
        )
        benchmark = HumanEvalBenchmark(task_ids=["HumanEval/0"])

        with TemporaryDirectory() as tmp:
            classroom = TeacherStudentClassroom(
                benchmark=benchmark,
                teacher=teacher,
                students=[student],
                output_path=Path(tmp) / "rounds.jsonl",
            )
            report = classroom.run(generations=1, tasks_per_generation=1)

            self.assertEqual(len(report.rounds), 1)
            self.assertEqual(report.average_before, 0.0)
            self.assertEqual(report.average_after, 1.0)
            self.assertTrue(student.harness.skills)
            self.assertTrue(student.harness.memory)
            self.assertTrue(teacher.harness.memory)
