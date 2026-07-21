#!/usr/bin/env python3
"""Faithful Reflexion text-effect gate for ALFWorld.

The initial attempt is shared. On an initial failure, compare two retries with
no memory against two retries with the same model's minimally prompted, full
reflection prepended. Results are resumable and one JSONL row is committed per
task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[5]
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for path in (ROOT, SKILLOPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

from research.steering.adapters.benchmarks import AlfworldAdapter  # noqa: E402
from research.steering.adapters.models.qwen import normalize_alfworld_action  # noqa: E402


SYSTEM = "You are an expert agent operating in the ALFRED Embodied Environment."
REFLECT_SYSTEM = (
    "You are the same agent reflecting on your own failed ALFWorld attempt. "
    "Write a concrete reflection for your next attempt."
)
TASK_TYPES = [
    "pick_and_place",
    "pick_two_obj_and_place",
    "look_at_obj_in_light",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_clean_then_place_in_recep",
]


def stable_seed(*parts: object) -> int:
    raw = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big")


def chat(model, tokenizer, device: str, system: str, user: str, *, seed: int,
         max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to(device)
    torch.manual_seed(seed)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature if temperature > 0 else None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def run_attempt(model, tokenizer, device: str, benchmark: AlfworldAdapter,
                entry: dict, memory: list[str], *,
                seed: int, temperature: float, max_steps: int,
                max_new_tokens: int) -> dict:
    gamefile = benchmark.local_gamefile(entry["gamefile"])
    env = None
    trace = []
    won = False
    try:
        env = benchmark.build(gamefile)
        obs, _ = env.reset({})
        for step in range(max_steps):
            observation = obs["text"][0]
            prompt = observation
            if memory:
                block = "\n\n".join(f"Reflection {i + 1}:\n{x}" for i, x in enumerate(memory))
                prompt = (
                    "## Reflections from your previous failed attempt(s)\n"
                    f"{block}\n\nUse these reflections in this retry.\n\n{prompt}"
                )
            raw_response = chat(
                model, tokenizer, device, SYSTEM, prompt,
                seed=stable_seed(seed, step), max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            response = normalize_alfworld_action(raw_response)
            obs, rewards, dones, infos = env.step([response])
            done = bool(dones[0].item() if hasattr(dones[0], "item") else dones[0])
            won = bool(infos[0].get("won", False)) if done and infos else False
            trace.append({
                "step": step,
                "observation": observation,
                "raw_response": raw_response,
                "response": response,
                "feedback": str(obs.get("anchor", [""])[0]),
                "reward": float(rewards[0]),
                "done": done,
            })
            if done:
                break
    except Exception as exc:  # keep a durable protocol/runtime boundary
        trace.append({"error": repr(exc)})
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return {"hard": int(won), "n_turns": len(trace), "trace": trace}


def reflection_prompt(entry: dict, attempt: dict, prior: list[str]) -> str:
    steps = []
    for row in attempt["trace"]:
        if "error" in row:
            steps.append(f"Runtime record: {row['error']}")
            continue
        steps.append(
            f"Step {row['step']}\nObservation: {row.get('observation', '')}\n"
            f"Your response: {row['response']}\n"
            f"Environment response: {row['feedback']}"
        )
    prior_text = "\n\n".join(prior)
    return (
        f"Task: {entry.get('task_description', '')}\n\n"
        + (f"Earlier reflection memory:\n{prior_text}\n\n" if prior else "")
        + "Failed attempt trajectory:\n"
        + "\n\n".join(steps)
        + "\n\nReflect on this specific failed attempt. Identify what went wrong and what "
          "you should do differently on the next attempt. Preserve concrete task state and "
          "action-order details. Do not turn it into a generic reusable skill. Return only the reflection."
    )


def select_tasks(manifest: dict, limit: int) -> list[str]:
    # Keep the compiler/reflexion gate on the frozen valid_seen selection split.
    # The manifest also contains valid_unseen test rows, which must remain held out.
    failed = [
        tid for tid in manifest["ids"]
        if tid.startswith("val:") and not manifest["tasks"][tid]["v0000"]["hard"]
    ]
    buckets = {kind: [] for kind in TASK_TYPES}
    for tid in sorted(failed):
        kind = manifest["tasks"][tid]["v0000"]["task_type"]
        buckets.setdefault(kind, []).append(tid)
    selected = []
    cursor = 0
    while len(selected) < limit:
        advanced = False
        for kind in TASK_TYPES:
            if cursor < len(buckets.get(kind, [])):
                selected.append(buckets[kind][cursor])
                advanced = True
                if len(selected) == limit:
                    break
        if not advanced:
            break
        cursor += 1
    random.Random(42).shuffle(selected)
    return selected


def load_groups(manifest: dict, plan_path: Path | None, splits: list[str], limit: int) -> list[dict]:
    if plan_path is None:
        return [
            {"group_id": task_id, "task_id": task_id, "task_seed": 0, "split": "legacy"}
            for task_id in select_tasks(manifest, limit)
        ]
    plan = json.loads(plan_path.read_text())
    groups = [group for group in plan["groups"] if group["split"] in set(splits)]
    if limit:
        groups = groups[:limit]
    missing = [group["task_id"] for group in groups if group["task_id"] not in manifest["tasks"]]
    if missing:
        raise ValueError(f"split plan contains unknown tasks: {missing[:3]}")
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=3)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "dev"])
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--reflection-tokens", type=int, default=512)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "research/steering/experiments/reflexion_T/resources/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    groups = load_groups(manifest, args.plan, args.splits, args.limit if args.plan is None else 0)
    (out_dir / "groups.json").write_text(json.dumps(groups, indent=2) + "\n")
    task_groups = groups[args.shard::args.num_shards]

    result_path = out_dir / f"results_shard{args.shard}.jsonl"
    done = set()
    if result_path.exists():
        done = {json.loads(line)["id"] for line in result_path.read_text().splitlines() if line.strip()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        device_map={"": args.device}, low_cpu_mem_usage=True,
    )
    model.eval()
    benchmark = AlfworldAdapter(SKILLOPT_ROOT / "data" / "alfworld_data", seed=42)

    with result_path.open("a") as fout:
        for index, group in enumerate(task_groups):
            group_id = group["group_id"]
            task_id = group["task_id"]
            task_seed = int(group["task_seed"])
            if group_id in done:
                continue
            started = time.time()
            entry = manifest["tasks"][task_id]["v0000"]
            initial = run_attempt(
                model, tokenizer, args.device, benchmark, entry, [],
                seed=stable_seed(group_id, task_seed, "initial"),
                temperature=args.temperature, max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
            )
            control = []
            reflex = []
            reflections = []
            if not initial["hard"]:
                # The no-memory control always receives the full two-retry
                # budget, independent of whether the Reflexion arm succeeds.
                for retry in range(2):
                    trial_seed = stable_seed(group_id, task_seed, "retry", retry)
                    control.append(run_attempt(
                        model, tokenizer, args.device, benchmark, entry, [], seed=trial_seed,
                        temperature=args.temperature, max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                    ))
                for retry in range(2):
                    trial_seed = stable_seed(group_id, task_seed, "retry", retry)
                    source = initial if retry == 0 else reflex[-1]
                    reflection = chat(
                        model, tokenizer, args.device, REFLECT_SYSTEM,
                        reflection_prompt(entry, source, reflections),
                        seed=stable_seed(group_id, task_seed, "reflection", retry),
                        max_new_tokens=args.reflection_tokens, temperature=args.temperature,
                    )
                    reflections.append(reflection)
                    reflex.append(run_attempt(
                        model, tokenizer, args.device, benchmark, entry, reflections, seed=trial_seed,
                        temperature=args.temperature, max_steps=args.max_steps,
                        max_new_tokens=args.max_new_tokens,
                    ))
                    if reflex[-1]["hard"]:
                        break
            row = {
                "id": group_id,
                "group_id": group_id,
                "task_id": task_id,
                "task_seed": task_seed,
                "split": group["split"],
                "task_type": entry["task_type"],
                "task_description": entry.get("task_description", ""),
                "initial": initial,
                "control_retries": control,
                "reflex_retries": reflex,
                "reflections": reflections,
                "initial_failed": int(not initial["hard"]),
                "control_any": int(any(x["hard"] for x in control)),
                "reflex_any": int(any(x["hard"] for x in reflex)),
                "elapsed_s": round(time.time() - started, 1),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            print(json.dumps({
                "shard": args.shard, "n": index, "id": group_id,
                "task_id": task_id, "split": group["split"],
                "initial": initial["hard"], "control_any": row["control_any"],
                "reflex_any": row["reflex_any"], "elapsed_s": row["elapsed_s"],
            }), flush=True)


if __name__ == "__main__":
    main()
