"""PocketFlow workflow nodes for one teacher-student classroom round.

PocketFlow is used only for orchestration. The research objects remain our own:
student/teacher agents, layered harnesses, HumanEval benchmark, and peer bank.
"""

from __future__ import annotations

try:
    from pocketflow import Flow, Node
except ImportError:  # pragma: no cover - optional fallback for minimal installs.
    Flow = None
    Node = object

from classroom.agent.student import StudentAgent
from classroom.agent.teacher import TeacherAgent
from classroom.agent.types import StudentAttempt
from classroom.benchmark.humaneval import CodeTask, HumanEvalBenchmark
from classroom.interaction.classroom import ClassroomRound
from classroom.interaction.peer import PeerExperienceBank


class AttemptNode(Node):
    def prep(self, shared: dict) -> dict:
        return shared

    def exec(self, shared: dict) -> dict:
        student: StudentAgent = shared["student"]
        task: CodeTask = shared["task"]
        benchmark: HumanEvalBenchmark = shared["benchmark"]
        completion, rationale = student.solve(task)
        evaluation = benchmark.evaluate(task, completion)
        return {
            "attempt": StudentAttempt(
                student_id=student.student_id,
                task_id=task.task_id,
                completion=completion,
                rationale=rationale,
                evaluation=evaluation,
                harness_snapshot=student.harness.to_dict(),
            ),
            "before": evaluation,
        }

    def post(self, shared: dict, prep_res: object, exec_res: dict) -> str:
        shared.update(exec_res)
        shared["peer_bank"].add_attempt(exec_res["attempt"])
        return "default"


class TeachNode(Node):
    def prep(self, shared: dict) -> dict:
        return shared

    def exec(self, shared: dict) -> dict:
        teacher: TeacherAgent = shared["teacher"]
        peer_bank: PeerExperienceBank = shared["peer_bank"]
        attempt: StudentAttempt = shared["attempt"]
        advice = teacher.teach(attempt, peer_summary=peer_bank.summarize_for_task(attempt.task_id))
        return {"advice": advice}

    def post(self, shared: dict, prep_res: object, exec_res: dict) -> str:
        shared.update(exec_res)
        shared["peer_bank"].add_advice(exec_res["advice"])
        return "default"


class ReviseNode(Node):
    def prep(self, shared: dict) -> dict:
        return shared

    def exec(self, shared: dict) -> dict:
        student: StudentAgent = shared["student"]
        task: CodeTask = shared["task"]
        benchmark: HumanEvalBenchmark = shared["benchmark"]
        student.apply_advice(shared["advice"])
        revised_completion, _ = student.solve(task)
        after = benchmark.evaluate(task, revised_completion)
        return {"after": after, "revised_completion": revised_completion}

    def post(self, shared: dict, prep_res: object, exec_res: dict) -> str:
        shared.update(exec_res)
        return "default"


class TeacherReflectNode(Node):
    def prep(self, shared: dict) -> dict:
        return shared

    def exec(self, shared: dict) -> ClassroomRound:
        teacher: TeacherAgent = shared["teacher"]
        teacher.observe_outcome(shared["advice"], shared["before"], shared["after"])
        return ClassroomRound(
            generation=shared["generation"],
            task_id=shared["task"].task_id,
            student_id=shared["student"].student_id,
            before=shared["before"],
            after=shared["after"],
            advice=shared["advice"].to_dict(),
            before_completion=shared["attempt"].completion,
            after_completion=shared["revised_completion"],
        )

    def post(self, shared: dict, prep_res: object, exec_res: ClassroomRound) -> str:
        shared["round_result"] = exec_res
        return "default"


def run_round_with_pocketflow(shared: dict) -> ClassroomRound:
    if Flow is None:
        raise RuntimeError("PocketFlow is not installed. Install `pocketflow` or use the direct classroom runner.")
    attempt = AttemptNode()
    teach = attempt >> TeachNode()
    revise = teach >> ReviseNode()
    revise >> TeacherReflectNode()
    Flow(start=attempt).run(shared)
    return shared["round_result"]
