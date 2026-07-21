from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class SerializedTrajectory:
    text: str
    step_char_spans: dict[int, tuple[int, int]]
    step_texts: dict[int, str]


def step_feedback(step: dict[str, Any]) -> str:
    return str(step.get("env_feedback") or step.get("observation") or step.get("feedback") or "").strip()


def step_decision(step: dict[str, Any]) -> str:
    return str(step.get("action") or step.get("model_response") or step.get("reasoning") or "").strip()


def serialize_raw_trajectory(task: str, trace: list[dict[str, Any]]) -> SerializedTrajectory:
    parts = [f"Task: {task.strip()}\n"]
    spans: dict[int, tuple[int, int]] = {}
    texts: dict[int, str] = {}
    cursor = len(parts[0])

    initial = str(trace[0].get("initial_observation") or "").strip() if trace else ""
    if initial:
        segment = f"Initial observation: {initial}\n"
        parts.append(segment)
        cursor += len(segment)

    for idx, step in enumerate(trace):
        prefix = f"\nAction {idx + 1}: "
        decision = step_decision(step)
        parts.append(prefix)
        cursor += len(prefix)
        start = cursor
        parts.append(decision)
        cursor += len(decision)
        spans[idx] = (start, cursor)
        texts[idx] = decision

        feedback = step_feedback(step)
        state = f"\nState {idx + 1}: {feedback}\n"
        parts.append(state)
        cursor += len(state)

    return SerializedTrajectory("".join(parts), spans, texts)


def tokenize_with_step_masks(tokenizer, serialized: SerializedTrajectory) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    encoded = tokenizer(
        serialized.text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"][0]
    masks: dict[int, torch.Tensor] = {}
    for step_id, (char_start, char_end) in serialized.step_char_spans.items():
        mask = (offsets[:, 1] > char_start) & (offsets[:, 0] < char_end) & (offsets[:, 1] > offsets[:, 0])
        if not bool(mask.any()):
            raise ValueError(f"empty token span for step {step_id}: {serialized.step_texts[step_id]!r}")
        masks[step_id] = mask
    return input_ids, masks


def extract_step_representations(
    model,
    tokenizer,
    *,
    task: str,
    trace: list[dict[str, Any]],
    layer: int,
    device: str,
) -> tuple[dict[int, torch.Tensor], SerializedTrajectory, dict[int, list[int]]]:
    serialized = serialize_raw_trajectory(task, trace)
    input_ids, masks = tokenize_with_step_masks(tokenizer, serialized)
    input_ids = input_ids.to(device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hidden = outputs.hidden_states[layer + 1][0].detach().float().cpu()

    reps: dict[int, torch.Tensor] = {}
    token_indices: dict[int, list[int]] = {}
    for step_id, mask in masks.items():
        indices = mask.nonzero(as_tuple=False).flatten()
        reps[step_id] = hidden[indices].mean(dim=0)
        token_indices[step_id] = indices.tolist()
    return reps, serialized, token_indices


def normalize(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.detach().float().flatten()
    return vector / vector.norm().clamp_min(1e-12)


def signed_pc1(vectors: torch.Tensor) -> torch.Tensor:
    vectors = vectors.detach().float()
    mean_direction = vectors.mean(dim=0)
    centered = vectors - mean_direction
    if vectors.shape[0] < 2 or float(centered.norm()) <= 1e-12:
        return normalize(mean_direction)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    vector = vh[0]
    if torch.dot(vector, mean_direction) < 0:
        vector = -vector
    return normalize(vector)


def centroid_cosine(positive: torch.Tensor, negative: torch.Tensor) -> float:
    return float(torch.dot(normalize(positive.mean(dim=0)), normalize(negative.mean(dim=0))))


def mean_pairwise_cosine(vectors: torch.Tensor) -> float:
    if vectors.shape[0] < 2:
        return float("nan")
    unit = vectors.float() / vectors.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
    similarity = unit @ unit.T
    mask = ~torch.eye(len(unit), dtype=torch.bool)
    return float(similarity[mask].mean())
