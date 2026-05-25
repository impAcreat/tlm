"""Layered harness state for teacher and student agents.

References:
- Reflexion/Voyager-style memory and skill accumulation for LLM agents.
- Agent-R1-style separation between the agent policy and environment feedback.
- Harness-level agent improvement: update prompts, memories, skills, and tool
  policies without changing base model weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PatchTarget = Literal["prompt", "skill", "memory", "tool_policy"]


@dataclass
class Skill:
    name: str
    content: str
    trigger: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "content": self.content, "trigger": self.trigger}


@dataclass
class MemoryItem:
    content: str
    source: str
    score_delta: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"content": self.content, "source": self.source, "score_delta": self.score_delta}


@dataclass
class ToolPolicy:
    allowed_tools: list[str] = field(default_factory=lambda: ["python_syntax_check"])
    max_tool_calls: int = 1
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "max_tool_calls": self.max_tool_calls,
            "notes": list(self.notes),
        }


@dataclass
class HarnessPatch:
    target: PatchTarget
    content: str
    reason: str
    source: str
    name: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "target": self.target,
            "content": self.content,
            "reason": self.reason,
            "source": self.source,
            "name": self.name,
        }


@dataclass
class HarnessState:
    """Mutable, inspectable agent harness."""

    identity: str
    base_prompt: str
    style: str
    skills: list[Skill] = field(default_factory=list)
    memory: list[MemoryItem] = field(default_factory=list)
    tool_policy: ToolPolicy = field(default_factory=ToolPolicy)
    patch_history: list[HarnessPatch] = field(default_factory=list)

    def render_system_prompt(self) -> str:
        skill_block = "\n".join(f"- {skill.name}: {skill.content}" for skill in self.skills) or "- none"
        memory_block = "\n".join(f"- {item.content}" for item in self.memory[-6:]) or "- none"
        tool_block = ", ".join(self.tool_policy.allowed_tools)
        return (
            f"{self.base_prompt}\n\n"
            f"Agent identity: {self.identity}\n"
            f"Preferred style: {self.style}\n\n"
            f"Available subtools: {tool_block}\n"
            f"Tool policy: max {self.tool_policy.max_tool_calls} tool calls before final answer.\n\n"
            f"Skills:\n{skill_block}\n\n"
            f"Memory:\n{memory_block}\n\n"
            "For coding tasks, return exactly one JSON object with fields:\n"
            '{"completion": "<Python code continuation>", "rationale": "<brief>"}'
        )

    def apply_patch(self, patch: HarnessPatch) -> None:
        if patch.target == "prompt":
            self.base_prompt = f"{self.base_prompt}\n{patch.content}"
        elif patch.target == "skill":
            self.skills.append(Skill(name=patch.name or f"skill_{len(self.skills) + 1}", content=patch.content))
        elif patch.target == "memory":
            self.memory.append(MemoryItem(content=patch.content, source=patch.source))
        elif patch.target == "tool_policy":
            self.tool_policy.notes.append(patch.content)
        else:
            raise ValueError(f"Unsupported patch target: {patch.target}")
        self.patch_history.append(patch)

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "base_prompt": self.base_prompt,
            "style": self.style,
            "skills": [skill.to_dict() for skill in self.skills],
            "memory": [item.to_dict() for item in self.memory],
            "tool_policy": self.tool_policy.to_dict(),
            "patch_history": [patch.to_dict() for patch in self.patch_history],
        }
