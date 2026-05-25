"""Student agent that solves tasks and updates its own harness."""

from __future__ import annotations

import json
import re

from classroom.agent.types import StudentFeature, TeachingAdvice
from classroom.benchmark.humaneval import CodeTask
from classroom.harness.state import HarnessPatch, HarnessState
from classroom.model.client import ChatModel


class StudentAgent:
    def __init__(self, student_id: str, feature: StudentFeature, model: ChatModel, harness: HarnessState) -> None:
        self.student_id = student_id
        self.feature = feature
        self.model = model
        self.harness = harness

    def solve(self, task: CodeTask) -> tuple[str, str]:
        messages = [
            {"role": "system", "content": self.harness.render_system_prompt()},
            {
                "role": "user",
                "content": (
                    f"Task {task.task_id}. Complete this Python function.\n"
                    "Return only JSON; the completion should be code that follows the prompt.\n\n"
                    f"{task.prompt}"
                ),
            },
        ]
        raw = self.model.complete(messages)
        return parse_student_response(raw)

    def apply_advice(self, advice: TeachingAdvice) -> None:
        own_memory = HarnessPatch(
            target="memory",
            content=f"On {advice.task_id}, teacher diagnosed: {advice.diagnosis}",
            reason="student self-reflection from teacher feedback",
            source=self.student_id,
            name="self_reflection",
        )
        self.harness.apply_patch(own_memory)
        for patch in advice.patches:
            self.harness.apply_patch(patch)


def parse_student_response(raw: str) -> tuple[str, str]:
    raw = raw.strip()
    for candidate in _student_response_candidates(raw):
        parsed = _parse_student_json(candidate)
        if parsed is not None:
            return parsed
    return _clean_completion(raw), ""


def _student_response_candidates(raw: str) -> list[str]:
    candidates = [raw]
    fenced = re.findall(r"```(?:json|python)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(item.strip() for item in fenced)
    extracted = _extract_json_objects(raw)
    candidates.extend(extracted)
    return [item for item in candidates if item]


def _extract_json_objects(text: str) -> list[str]:
    objects: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                objects.append(text[start : index + 1])
                start = None
    return objects


def _parse_student_json(candidate: str) -> tuple[str, str] | None:
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    completion = data.get("completion", data.get("code", ""))
    rationale = data.get("rationale", data.get("explanation", ""))
    return _clean_completion(str(completion)), str(rationale)


def _clean_completion(completion: str) -> str:
    completion = completion.strip("\r\n")
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)```", completion, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        completion = fenced.group(1).strip("\r\n")
    return completion + ("\n" if completion and not completion.endswith("\n") else "")
