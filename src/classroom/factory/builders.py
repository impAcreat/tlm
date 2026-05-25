"""Factories for teacher/student agents with layered harnesses."""

from __future__ import annotations

from classroom.agent.student import StudentAgent
from classroom.agent.teacher import TeacherAgent
from classroom.agent.types import StudentFeature
from classroom.harness.state import HarnessState, Skill, ToolPolicy
from classroom.model.client import ChatModel, OpenAICompatibleChatModel


def make_chat_model(
    model_name: str,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    timeout_s: int = 60,
    max_tokens: int | None = None,
    enable_thinking: bool | None = None,
) -> OpenAICompatibleChatModel:
    return OpenAICompatibleChatModel(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout_s=timeout_s,
        max_tokens=max_tokens,
        enable_thinking=enable_thinking,
    )


def make_teacher(model: ChatModel | None) -> TeacherAgent:
    harness = HarnessState(
        identity="teacher",
        base_prompt=(
            "You are a teacher agent for code-solving students. Diagnose failures, "
            "write actionable harness patches, and prefer patches that transfer across tasks."
        ),
        style="diagnostic, concise, evidence-driven",
        skills=[
            Skill(
                name="grade_then_patch",
                content="Use execution results to propose prompt, skill, memory, or tool-policy patches.",
            )
        ],
        tool_policy=ToolPolicy(allowed_tools=["grade_attempt", "propose_harness_patch"], max_tool_calls=2),
    )
    return TeacherAgent(teacher_id="teacher_0", model=model, harness=harness)


def make_students(model: ChatModel, count: int = 3) -> list[StudentAgent]:
    features = [
        StudentFeature(
            name="planner",
            style="plan before coding, then write compact code",
            bias="strong decomposition, may overthink simple tasks",
        ),
        StudentFeature(
            name="edge_case_solver",
            style="prioritize boundary cases and examples",
            bias="careful verification, may be verbose",
        ),
        StudentFeature(
            name="minimalist",
            style="write the shortest correct implementation",
            bias="fast direct answers, may miss hidden edge cases",
        ),
    ]
    students: list[StudentAgent] = []
    for index, feature in enumerate(features[:count]):
        harness = HarnessState(
            identity=f"student_{index}_{feature.name}",
            base_prompt=(
                "You are a student coding agent. Solve the function completion task. "
                "Use the harness skills and memory as your evolving study notes."
            ),
            style=f"{feature.style}. Known bias: {feature.bias}.",
            skills=[
                Skill(
                    name="humaneval_completion_format",
                    content="Return only the code continuation after the given prompt, not a full standalone file.",
                )
            ],
            tool_policy=ToolPolicy(allowed_tools=["python_syntax_check"], max_tool_calls=1),
        )
        students.append(StudentAgent(student_id=harness.identity, feature=feature, model=model, harness=harness))
    return students
