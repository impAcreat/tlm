#!/usr/bin/env python3
"""Dev-only causal calibration of layers and hidden-relative steering dose."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for path in (ROOT, SKILLOPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

from research.steering.adapters.benchmarks import AlfworldAdapter  # noqa: E402
from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402
from research.steering.adapters.models.qwen import normalize_alfworld_action  # noqa: E402
from research.steering.experiments.reflexion_T.experimental.hidden_relative import (  # noqa: E402
    HiddenRelativeSteerer,
)

SYSTEM = "You are an expert agent operating in the ALFRED Embodied Environment."
ACTION_RE = re.compile(r"<action>(.*?)</action>", re.S | re.I)


def load_records(paths: list[Path]) -> list[dict]:
    merged = {}
    for path in paths:
        shard = torch.load(path, weights_only=False)
        overlap = set(merged) & set(shard)
        if overlap:
            raise ValueError(f"duplicate units: {sorted(overlap)[:3]}")
        merged.update(shard)
    return list(merged.values())


def select_dev_units(records: list[dict], limit: int) -> list[dict]:
    eligible = [r for r in records if r.get("split") == "dev" and r.get("text_success")]
    eligible.sort(key=lambda r: (not bool(r.get("paired_effective")), r["task_id"], r["retry_index"]))
    selected, used_tasks = [], set()
    for record in eligible:
        if record["task_id"] in used_tasks:
            continue
        selected.append(record)
        used_tasks.add(record["task_id"])
        if limit and len(selected) >= limit:
            break
    return selected


def chat(model, tokenizer, device: str, prompt: str, steerer, max_new_tokens: int) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)
    generate_kwargs = dict(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with steerer if steerer is not None else contextlib.nullcontext():
        output = model.generate(**generate_kwargs)
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True).strip()


def action_text(response: str) -> str:
    match = ACTION_RE.search(response)
    return match.group(1).strip().lower() if match else ""


def run_arm(model, tokenizer, device, benchmark, entry, *, reflection, steerer_spec,
            max_steps, max_new_tokens):
    env = None
    trace, actions = [], []
    won = False
    try:
        env = benchmark.build(benchmark.local_gamefile(entry["gamefile"]))
        obs, _ = env.reset({})
        for step in range(max_steps):
            prompt = obs["text"][0]
            if reflection:
                prompt = (
                    "## Reflections from your previous failed attempt(s)\n"
                    f"Reflection 1:\n{reflection.strip()}\n\n"
                    f"Use these reflections in this retry.\n\n{prompt}"
                )
            steerer = HiddenRelativeSteerer(model=model, **steerer_spec) if steerer_spec else None
            raw = chat(model, tokenizer, device, prompt, steerer, max_new_tokens)
            repaired = ACTION_RE.search(raw) is None
            response = normalize_alfworld_action(raw)
            action = action_text(response)
            actions.append(action)
            obs, rewards, dones, infos = env.step([response])
            done = bool(dones[0].item() if hasattr(dones[0], "item") else dones[0])
            won = bool(infos[0].get("won", False)) if done and infos else False
            trace.append({
                "step": step,
                "action": action,
                "raw_response": raw,
                "response": response,
                "feedback": str(obs.get("anchor", [""])[0]),
                "format_repaired": repaired,
                "reward": float(rewards[0]),
                "done": done,
            })
            if done:
                break
    except Exception as exc:
        trace.append({"error": repr(exc)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    repeats = sum(a and a == b for a, b in zip(actions[1:], actions[:-1]))
    return {
        "hard": int(won),
        "n_turns": len(trace),
        "format_repair_rate": sum(bool(x.get("format_repaired")) for x in trace) / max(1, len(trace)),
        "repeat_rate": repeats / max(1, len(actions) - 1),
        "runtime_error": next((x["error"] for x in trace if "error" in x), None),
        "actions": actions,
        "trace": trace,
    }


def random_direction(vector: torch.Tensor, unit_id: str, layer: int) -> torch.Tensor:
    seed = int.from_bytes(hashlib.sha256(f"{unit_id}|{layer}".encode()).digest()[:8], "big")
    generator = torch.Generator().manual_seed(seed)
    random = torch.randn(vector.shape, generator=generator)
    return random / random.norm().clamp_min(1e-12)


def mismatch_donor(record: dict, records: list[dict]) -> dict:
    """Choose a deterministic different-task donor for semantic specificity."""
    candidates = sorted(
        (row for row in records if row["task_id"] != record["task_id"]),
        key=lambda row: row["unit_id"],
    )
    if not candidates:
        raise ValueError("mismatched control requires at least two distinct tasks")
    index = int.from_bytes(hashlib.sha256(record["unit_id"].encode()).digest()[:8], "big")
    return candidates[index % len(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--multipliers", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--limit-units", type=int, default=12)
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    shard_index, shard_count = map(int, args.shard.split("/"))
    all_records = select_dev_units(load_records(args.inputs), args.limit_units)
    records = all_records[shard_index::shard_count]
    manifest = json.loads(args.manifest.read_text())
    model, tokenizer = load_causal_lm(args.model_path, args.device)
    benchmark = AlfworldAdapter(SKILLOPT_ROOT / "data" / "alfworld_data", seed=42)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        done = {json.loads(line)["eval_id"] for line in args.output.read_text().splitlines() if line.strip()}
    with args.output.open("a") as stream:
        for record in records:
            entry = manifest["tasks"][record["task_id"]]["v0000"]
            specs = [("baseline", -1, 0.0), ("text", -1, 0.0)]
            specs += [
                (arm, layer, multiplier)
                for layer in args.layers
                for multiplier in args.multipliers
                for arm in ("extracted", "random", "mismatched")
            ]
            for arm, layer, multiplier in specs:
                eval_id = f"{record['unit_id']}|{arm}|L{layer}|m{multiplier:g}"
                if eval_id in done:
                    continue
                vector = record["vector"][layer].float() if layer >= 0 else None
                donor = None
                if arm == "random":
                    vector = random_direction(vector, record["unit_id"], layer)
                elif arm == "mismatched":
                    donor = mismatch_donor(record, all_records)
                    vector = donor["vector"][layer].float()
                steerer_spec = None
                if vector is not None:
                    steerer_spec = {
                        "layer": layer,
                        "vector": vector,
                        "natural_rho": float(record["natural_rho_median"][layer]),
                        "multiplier": multiplier,
                        "min_addition_norm": float(record["delta_norm_q10"][layer]),
                        "max_addition_norm": float(record["delta_norm_q90"][layer]),
                    }
                started = time.time()
                result = run_arm(
                    model, tokenizer, args.device, benchmark, entry,
                    reflection=record["text"] if arm == "text" else None,
                    steerer_spec=steerer_spec,
                    max_steps=args.max_steps,
                    max_new_tokens=args.max_new_tokens,
                )
                row = {
                    "eval_id": eval_id,
                    "unit_id": record["unit_id"],
                    "task_id": record["task_id"],
                    "group_id": record["group_id"],
                    "arm": arm,
                    "layer": layer,
                    "multiplier": multiplier,
                    "text_success_at_collection": bool(record["text_success"]),
                    "paired_effective_at_collection": bool(record["paired_effective"]),
                    "mismatch_donor_unit_id": donor["unit_id"] if donor else None,
                    "mismatch_donor_task_id": donor["task_id"] if donor else None,
                    "elapsed_s": round(time.time() - started, 1),
                    **result,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(json.dumps({k: row[k] for k in ("eval_id", "hard", "n_turns", "format_repair_rate", "repeat_rate", "runtime_error")}), flush=True)


if __name__ == "__main__":
    main()
