"""Extract Qwen3.5-4B hidden-state representations for SkillOpt selection rollouts.

Behavior-only serialization (task + action/state trace, no skill text, no
reward/done markers), exact action-token spans per step, mean pooling.

Per trajectory we store, in float16:
  step_reps  [n_steps, len(STEP_LAYERS), 2560]  action-span mean at selected layers
  traj_all   [32, 2560]  mean over per-step means, all layers
  early5     [32, 2560]  mean over first <=5 step means
  last_step  [32, 2560]  final step action-span mean
Layer convention matches prior runs: layer L reads outputs.hidden_states[L+1].
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/data5/ninghan/tlm")
sys.path.insert(0, str(ROOT))

from research.steering.core.hf import load_causal_lm  # noqa: E402
from research.steering.skill_edit.intra_step_mean import (  # noqa: E402
    serialize_raw_trajectory,
    tokenize_with_step_masks,
)

STEP_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
NUM_LAYERS = 32
MAX_TOKENS = 14000


def step_flags(trace: list[dict]) -> dict[str, list[bool]]:
    seen: set[str] = set()
    invalid, repeat = [], []
    for step in trace:
        action = str(step.get("action") or "").strip().lower()
        feedback = str(step.get("env_feedback") or "").strip().lower()
        invalid.append("nothing happens" in feedback or not feedback)
        repeat.append(bool(action) and action in seen)
        if action:
            seen.add(action)
    return {"invalid": invalid, "repeat": repeat}


@torch.no_grad()
def extract_one(model, tokenizer, device: str, task: str, trace: list[dict]) -> dict:
    serialized = serialize_raw_trajectory(task, trace)
    input_ids, masks = tokenize_with_step_masks(tokenizer, serialized)
    truncated = False
    if input_ids.shape[1] > MAX_TOKENS:
        # drop trailing steps until within budget
        keep = max(k for k in masks if int(masks[k].nonzero().max()) < MAX_TOKENS)
        trace = trace[: keep + 1]
        serialized = serialize_raw_trajectory(task, trace)
        input_ids, masks = tokenize_with_step_masks(tokenizer, serialized)
        truncated = True
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), output_hidden_states=True)
    # hidden_states: tuple of 33 tensors [1, seq, 2560]
    hs = torch.stack([h[0] for h in out.hidden_states[1:]], dim=0).float()  # [32, seq, d]

    n_steps = len(masks)
    step_means = torch.zeros(n_steps, NUM_LAYERS, hs.shape[-1])
    for sid in range(n_steps):
        idx = masks[sid].nonzero(as_tuple=False).flatten().to(hs.device)
        step_means[sid] = hs[:, idx, :].mean(dim=1)

    traj_all = step_means.mean(dim=0)
    early5 = step_means[: min(5, n_steps)].mean(dim=0)
    last_step = step_means[-1]
    return {
        "step_reps": step_means[:, STEP_LAYERS, :].half(),
        "traj_all": traj_all.half(),
        "early5": early5.half(),
        "last_step": last_step.half(),
        "n_steps": n_steps,
        "n_tokens": int(input_ids.shape[1]),
        "truncated": truncated,
        "flags": step_flags(trace),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conditions", default="v0000,step1,step2")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    reps_dir = out_dir / "reps"
    reps_dir.mkdir(exist_ok=True)

    model, tokenizer = load_causal_lm(args.model_path, args.device)

    meta = {"step_layers": STEP_LAYERS, "num_layers": NUM_LAYERS, "model_path": args.model_path,
            "pooling": "action-span mean per step; traj = mean of step means", "dtype": "float16"}
    (reps_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for cond in args.conditions.split(","):
        out_path = reps_dir / f"{cond}.pt"
        if out_path.exists():
            print(f"[skip] {out_path} exists", flush=True)
            continue
        store: dict[str, dict] = {}
        t0 = time.time()
        for i, tid in enumerate(manifest["ids"]):
            entry = manifest["tasks"][tid][cond]
            trace = json.loads(Path(entry["conversation"]).read_text())
            if not trace:
                print(f"[warn] empty trace {cond}/{tid}", flush=True)
                continue
            store[tid] = extract_one(model, tokenizer, args.device, entry["task_description"], trace)
            store[tid]["hard"] = entry["hard"]
            store[tid]["task_type"] = entry["task_type"]
            if (i + 1) % 20 == 0:
                print(f"[{cond}] {i + 1}/{len(manifest['ids'])} elapsed {time.time() - t0:.0f}s", flush=True)
        torch.save(store, out_path)
        print(f"[done] {cond}: {len(store)} trajectories -> {out_path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
