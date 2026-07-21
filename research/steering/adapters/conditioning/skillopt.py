"""SkillOpt text manipulation adapter with a lazy optional dependency."""
from __future__ import annotations


def skill_prompt(content: str) -> str:
    if not content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{content}\n"
    )


def apply_skillopt_edit(base_skill: str, edit: dict) -> str:
    from skillopt.optimizer.skill import apply_edit

    return apply_edit(base_skill, edit)
