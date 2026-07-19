from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from research.steering.core.hooks import ActivationSteerer


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "For every turn, state one explicit <think>...</think> reasoning block before "
    "the required <action>...</action>. Choose exactly one admissible action."
)


@dataclass(frozen=True)
class SteeringSpec:
    layer: int
    vector: torch.Tensor
    alpha: float
    token_slice: slice | None = None


@dataclass(frozen=True)
class RolloutStep:
    step: int
    action: str | None
    response: str
    observation: str
    reward: float
    done: bool
    valid: bool
    prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_action(model_response: str) -> str | None:
    match = re.search(r"<action>(.*?)</action>", model_response or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


def has_think_tag(model_response: str) -> bool:
    text = model_response or ""
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL | re.IGNORECASE))


def normalize_model_response(model_response: str) -> str:
    response = (model_response or "").strip()
    action = extract_action(response)
    if not response or action is None:
        return "<think>missing action tag</think><action>look</action>"
    if not has_think_tag(response):
        return f"<think>missing think tag</think>{response}"
    return response


def build_skill_prompt(skill_content: str) -> str:
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content.strip()}\n"
    )


def build_messages(user_prompt: str, skill_content: str = "") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{build_skill_prompt(skill_content)}{user_prompt}"},
    ]


def load_steering_vector(
    path: str | Path,
    *,
    alpha: float,
    layer: int | None = None,
    token_slice: slice | None = None,
) -> SteeringSpec:
    artifact = torch.load(Path(path), map_location="cpu")
    if isinstance(artifact, dict):
        vector = artifact.get("vector")
        inferred_layer = artifact.get("layer")
    else:
        vector = artifact
        inferred_layer = None
    if vector is None:
        raise ValueError(f"no 'vector' tensor found in {path}")
    vector = vector.detach().float().flatten()
    norm = vector.norm()
    if not torch.isfinite(norm) or float(norm) <= 0.0:
        raise ValueError(f"degenerate steering vector in {path}")
    chosen_layer = int(layer if layer is not None else inferred_layer)
    return SteeringSpec(layer=chosen_layer, vector=vector / norm, alpha=float(alpha), token_slice=token_slice)


def summarize_steps(steps: list[RolloutStep]) -> dict[str, Any]:
    repeated = 0
    previous: str | None = None
    for step in steps:
        action = (step.action or "").strip().lower()
        if action and previous == action:
            repeated += 1
        previous = action or previous
    return {
        "n_turns": len(steps),
        "invalid_actions": sum(1 for step in steps if not step.valid),
        "repeated_actions": repeated,
        "total_reward": float(sum(step.reward for step in steps)),
    }


def chat_input_ids(tokenizer, messages: list[dict[str, str]], *, enable_thinking: bool | None = None) -> torch.Tensor:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        kwargs = {
            "conversation": messages,
            "add_generation_prompt": True,
            "return_tensors": "pt",
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            rendered = tokenizer.apply_chat_template(**kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered = tokenizer.apply_chat_template(**kwargs)
        if isinstance(rendered, torch.Tensor):
            return rendered
        if hasattr(rendered, "input_ids"):
            return rendered.input_ids
        text_kwargs = dict(kwargs)
        text_kwargs.pop("return_tensors", None)
        text_kwargs["tokenize"] = False
        rendered_text = tokenizer.apply_chat_template(**text_kwargs)
        return tokenizer(rendered_text, return_tensors="pt", add_special_tokens=False).input_ids
    text = "\n\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"
    return tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids


def generate_response(
    model,
    tokenizer,
    messages: list[dict[str, str]],
    *,
    device: str,
    steering: SteeringSpec | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    enable_thinking: bool | None = None,
) -> str:
    input_ids = chat_input_ids(tokenizer, messages, enable_thinking=enable_thinking).to(device)
    attention_mask = torch.ones_like(input_ids)
    do_sample = temperature > 0.0
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        if steering is None or steering.alpha == 0.0:
            output_ids = model.generate(**gen_kwargs)
        else:
            with ActivationSteerer(
                model,
                layer=steering.layer,
                vector=steering.vector,
                alpha=steering.alpha,
                token_slice=steering.token_slice,
            ):
                output_ids = model.generate(**gen_kwargs)
    new_ids = output_ids[:, input_ids.shape[1]:]
    return tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()
