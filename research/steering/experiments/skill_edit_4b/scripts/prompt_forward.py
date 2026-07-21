"""Prompt-conditioned skill contrast: identical replayed states, three skill
prompts (bad = rough_v1, good = step-2 best, none), last-token hidden states.

Deltas per state and layer:
  good_minus_bad, good_minus_none, bad_minus_none

Output: prompt_deltas.pt  {deltas: {name: [n_states, 32, 2560] fp16},
                           states: [meta...], mean_vectors: {name: [32, 2560]}}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."


def build_skill_prompt(skill_content: str) -> str:
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


@torch.no_grad()
def last_token_layers(model, tokenizer, device: str, user_prompt: str) -> torch.Tensor:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                            return_tensors="pt", enable_thinking=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states[1:]], dim=0).float().cpu()  # [32, 2560]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompts", default="prompts_v0000.jsonl")
    parser.add_argument("--good-skill-path", default=str(RUN_ROOT / "steps/step_0002/candidate_skill.md"),
                        help="step-2 candidate by default == the skill behind the 62.14% rollouts")
    parser.add_argument("--out-name", default="prompt_deltas.pt")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    skills = {
        "bad": (RUN_ROOT / "skills/skill_v0000.md").read_text(),
        "good": Path(args.good_skill_path).read_text(),
        "none": "",
    }

    states = []
    with (out_dir / args.prompts).open() as f:
        for line in f:
            if line.strip():
                states.append(json.loads(line))
    print(f"{len(states)} states", flush=True)

    model, tokenizer = load_causal_lm(args.model_path, args.device)

    reps = {k: [] for k in skills}
    t0 = time.time()
    for i, st in enumerate(states):
        for name, content in skills.items():
            user = build_skill_prompt(content) + "\n" + st["obs_text"] if content else st["obs_text"]
            reps[name].append(last_token_layers(model, tokenizer, args.device, user).half())
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(states)} elapsed {time.time() - t0:.0f}s", flush=True)

    reps = {k: torch.stack(v) for k, v in reps.items()}  # [n, 32, 2560]
    deltas = {
        "good_minus_bad": (reps["good"].float() - reps["bad"].float()).half(),
        "good_minus_none": (reps["good"].float() - reps["none"].float()).half(),
        "bad_minus_none": (reps["bad"].float() - reps["none"].float()).half(),
    }
    mean_vectors = {k: v.float().mean(dim=0).half() for k, v in deltas.items()}
    meta = [{k: st[k] for k in ("task_id", "step", "category", "task_type", "hard")} for st in states]
    torch.save({"deltas": deltas, "states": meta, "mean_vectors": mean_vectors,
                "reps_last_token": {k: v for k, v in reps.items()},
                "good_skill_path": args.good_skill_path},
               out_dir / args.out_name)
    print(f"saved {args.out_name} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
