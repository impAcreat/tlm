"""Deterministic feedback-quality checks shared by EFM gates and diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from .models import StepFeedback


ENTITY = re.compile(r"\b([a-z]{3,}) (\d+)\b")
STOP_ENTITY = {"step"}
TRANSFORM = ["cleaned", "cooled", "heated", "sliced"]
OBS_VERB = {
    "clean": "cleaned",
    "cool": "cooled",
    "heat": "heated",
    "slice": "sliced",
    "open": "opened",
    "close": "closed",
    "pick up": "picked up",
    "take": "took",
    "move": "moved",
    "put": "put",
    "place": "placed",
    "arrive": "arrived",
}
META = re.compile(
    r"raw[_ ]observation|supplied data|step[_ ]?id|recent[_ ]actions|"
    r"in step \d+|the observation confirms|the task('s| ) ",
    re.I,
)
CONCRETE_OBS = re.compile(
    r"you (arrive|open|close|pick up|take|move|put|clean|cool|heat|see)|"
    r"is (open|closed)|you see nothing|not carrying anything",
    re.I,
)
TASK_TARGET = re.compile(
    r"\b(?:examine|find|put|pick up|clean|cool|heat|slice|move|take)\s+the\s+([a-z]+)\b",
    re.I,
)
ABSENCE_WORDS = re.compile(r"\b(not visible|not present|not observed|absent|missing|no\b|nothing)\b", re.I)


@dataclass(frozen=True)
class FeedbackQuality:
    consistency: int
    completeness: int
    efficiency: int
    issues: list[str]

    def as_tuple(self) -> tuple[int, int, int]:
        return self.consistency, self.completeness, self.efficiency


def _feedback_from(value: StepFeedback | dict[str, Any]) -> StepFeedback:
    if isinstance(value, StepFeedback):
        return value
    return StepFeedback(
        core_signal=str(value.get("core_signal", "")),
        signal_type=str(value.get("signal_type", "ambiguity")),  # type: ignore[arg-type]
        filtered_out=str(value.get("filtered_out", "")),
        fallback=bool(value.get("fallback", False)),
    )


def entities(text: str) -> set[str]:
    return {f"{name} {digit}" for name, digit in ENTITY.findall(str(text).lower()) if name not in STOP_ENTITY}


def negated(text: str, entity: str) -> bool:
    low = str(text).lower()
    for match in re.finditer(re.escape(entity), low):
        window = low[max(0, match.start() - 30):match.end() + 20]
        if not re.search(
            r"\bno\b|\bnot\b|n't|absent|without|no longer|"
            r"is not|are not|not present|not observed|not visible",
            window,
        ):
            return False
    return True


def current_verbs(action: Any, observation: Any) -> set[str]:
    low = f"{action} {observation}".lower()
    return {normalized for raw, normalized in OBS_VERB.items() if raw in low}


def task_target(task_description: Any) -> str:
    match = TASK_TARGET.search(str(task_description or ""))
    return match.group(1).lower() if match else ""


def mentions_target_absence(feedback_text: str, target: str) -> bool:
    text = str(feedback_text).lower()
    if target not in text:
        return False
    if ABSENCE_WORDS.search(text):
        return True
    return bool(re.search(rf"\b{re.escape(target)}\b.{{0,40}}\b(is|are)\s+not\b", text))


def consistency(feedback_text: str, action: Any, observation: Any, recent: list[str] | None = None) -> tuple[int, list[str]]:
    del recent
    observation_entities = entities(str(observation))
    verbs = current_verbs(action, observation)
    feedback = str(feedback_text).lower()
    issues: list[str] = []

    for entity in entities(str(feedback_text)):
        if entity in observation_entities or negated(str(feedback_text), entity):
            continue
        issues.append(f"entity '{entity}' not in obs")

    for verb in TRANSFORM:
        if verb not in feedback or verb in verbs:
            continue
        if re.search(r"(no|not|n't)\s+\w*\s*" + re.escape(verb), feedback):
            continue
        issues.append(f"transform '{verb}' not in current step")

    if "not carrying anything" in str(observation).lower():
        positive_carry = re.search(r"\b(carrying|holding|in .{0,12}possession)\b", feedback)
        denies = re.search(r"not\s+carry|isn't\s+carry|no\s+items|nothing", feedback)
        if positive_carry and not denies:
            issues.append("claims carrying but obs says not carrying")

    return (1 if not issues else 0), issues


def completeness(
    signal_type: str,
    observation: Any,
    *,
    feedback_text: str = "",
    task_description: Any = "",
) -> int:
    if CONCRETE_OBS.search(str(observation)) and signal_type == "ambiguity":
        return 0
    target = task_target(task_description)
    if target and re.search(r"\byou (arrive|see)|\bon the\b", str(observation), re.I):
        if not re.search(rf"\b{re.escape(target)}\s+\d+\b", str(observation).lower()):
            if not mentions_target_absence(feedback_text, target):
                return 0
    return 1


def efficiency(feedback_text: str) -> int:
    text = str(feedback_text)
    if len(text) > 180:
        return 0
    if META.search(text):
        return 0
    return 1


def score_feedback(
    feedback: StepFeedback | dict[str, Any],
    *,
    action: Any,
    raw_observation: Any,
    recent_actions: list[str] | None = None,
    task_description: Any = "",
) -> FeedbackQuality:
    item = _feedback_from(feedback)
    cons, issues = consistency(item.core_signal, action, raw_observation, recent_actions or [])
    return FeedbackQuality(
        consistency=cons,
        completeness=completeness(
            item.signal_type,
            raw_observation,
            feedback_text=item.core_signal,
            task_description=task_description,
        ),
        efficiency=efficiency(item.core_signal),
        issues=issues,
    )
