"""Causal steering evaluation with skillopt-faithful prompting.

Runs HF greedy ALFWorld rollouts under a chosen skill prompt, optionally adding
a steering vector at one layer (all token positions), and reports success.

Arms are selected via CLI so multiple tmux jobs can run in parallel on
different GPUs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[5]
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for p in (str(ROOT), str(SKILLOPT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

from research.steering.adapters.models.loading import load_causal_lm  # noqa: E402
from research.steering.adapters.models.hf_steering import ActivationSteerer, MultiLayerActivationSteerer  # noqa: E402
from skillopt.envs.alfworld.rollout import build_alfworld_env  # noqa: E402

RUN_ROOT = ROOT / "benchmarks/skillopt/outputs/skillopt_clean_repro_seed42_20260714"
SYSTEM_PROMPT = "You are an expert agent operating in the ALFRED Embodied Environment."


class GenOnlySteerer(ActivationSteerer):
    """Steer only decode-step forwards (seq_len == 1), leaving prefill intact."""

    def _steer_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        return super()._steer_hidden(hidden)


class GenOnlyMultiLayerSteerer(MultiLayerActivationSteerer):
    """Multi-layer variant of GenOnlySteerer."""

    def _steer_hidden(self, hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if hidden.shape[1] > 1:
            return hidden
        return super()._steer_hidden(hidden, vector)

SKILLS = {
    "bad": RUN_ROOT / "skills/skill_v0000.md",
    "good": RUN_ROOT / "steps/step_0002/candidate_skill.md",
    "none": None,
}


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
def generate(model, tokenizer, device, user_prompt, steerer_args, max_new_tokens,
             temperature=0.0, gen_only=False):
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
    kwargs = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  max_new_tokens=max_new_tokens, do_sample=temperature > 0.0,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                  eos_token_id=tokenizer.eos_token_id)
    if temperature > 0.0:
        kwargs["temperature"] = temperature
    if steerer_args is None:
        out = model.generate(**kwargs)
    elif "vectors" in steerer_args:
        cls = GenOnlyMultiLayerSteerer if gen_only else MultiLayerActivationSteerer
        with cls(model, **steerer_args):
            out = model.generate(**kwargs)
    else:
        cls = GenOnlySteerer if gen_only else ActivationSteerer
        with cls(model, **steerer_args):
            out = model.generate(**kwargs)
    return tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--skill", choices=list(SKILLS), default="bad")
    parser.add_argument("--skill-path", default="", help="explicit skill file path; overrides --skill")
    parser.add_argument("--vector-path", default="")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--random-vector", action="store_true",
                        help="replace the loaded vector with a random unit vector (seed 7)")
    parser.add_argument("--multi-vector-path", default="",
                        help="gmb_raw_means.pt-style file; inject raw mean deltas at --multi-layers")
    parser.add_argument("--multi-layers", default="6-22", help="inclusive layer range lo-hi")
    parser.add_argument("--gen-only", action="store_true",
                        help="steer only decode steps, leave the prompt encoding untouched")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample-tag", default="", help="tag + rng seed offset for sampled arms")
    parser.add_argument("--vector-map", default="",
                        help="JSON {task_id: {path, alpha}}; per-task vector, overrides --vector-path")
    parser.add_argument("--skill-map", default="",
                        help="JSON {task_id: skill_path}; per-task skill text, overrides --skill-path")
    parser.add_argument("--model-path", default=str(ROOT / "models/Qwen3.5-4B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--categories", default="repaired,both_fail")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--task-ids", default="", help="comma-separated explicit ids override")
    parser.add_argument("--task-file", default="", help="file with one task id per line; overrides categories")
    parser.add_argument("--max-steps", type=int, default=35)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    cats = manifest["pairs"]["v0000_vs_step2"]

    if args.task_file:
        tids = [l.strip() for l in Path(args.task_file).read_text().splitlines() if l.strip()]
        if args.limit:
            tids = tids[: args.limit]
    elif args.task_ids:
        tids = args.task_ids.split(",")
    else:
        tids = []
        for cat in args.categories.split(","):
            tids.extend(cats[cat])
        tids = sorted(tids)
        if args.limit:
            tids = tids[: args.limit]

    if args.skill_path:
        skill_content = Path(args.skill_path).read_text()
    else:
        skill_content = SKILLS[args.skill].read_text() if SKILLS[args.skill] else ""
    skill_prefix = build_skill_prompt(skill_content)

    vector_map = json.loads(Path(args.vector_map).read_text()) if args.vector_map else None
    skill_map = json.loads(Path(args.skill_map).read_text()) if args.skill_map else None

    steerer_args = None
    if args.multi_vector_path:
        art = torch.load(args.multi_vector_path, weights_only=False)
        means = art["mean_delta"].float()  # [32, 2560]
        if "-" in args.multi_layers:
            lo, hi = (int(x) for x in args.multi_layers.split("-"))
            layer_list = list(range(lo, hi + 1))
        else:
            layer_list = [int(x) for x in args.multi_layers.split(",")]
        vectors = {L: means[L] for L in layer_list}
        if args.random_vector:
            g = torch.Generator().manual_seed(7)
            vectors = {L: torch.randn(means[L].shape, generator=g) * means[L].norm() / (2560 ** 0.5)
                       for L in vectors}
            vectors = {L: v * means[L].norm() / v.norm() for L, v in vectors.items()}
        steerer_args = dict(vectors=vectors, alpha=args.alpha, token_slice=None)
    elif args.vector_path:
        art = torch.load(args.vector_path, weights_only=False)
        vector = art["vector"].float().flatten()
        vector = vector / vector.norm().clamp_min(1e-12)
        layer = args.layer if args.layer >= 0 else int(art["layer"])
        if args.random_vector:
            g = torch.Generator().manual_seed(7)
            vector = torch.randn(vector.shape, generator=g)
            vector = vector / vector.norm()
        steerer_args = dict(layer=layer, vector=vector, alpha=args.alpha, token_slice=None)

    model, tokenizer = load_causal_lm(args.model_path, args.device)

    arm_dir = out_dir / "steered_eval" / args.arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)
    results_path = arm_dir / "results.jsonl"
    done = set()
    if results_path.exists():
        with results_path.open() as f:
            for line in f:
                if line.strip():
                    done.add(json.loads(line)["id"])

    with results_path.open("a") as fout:
        for n, tid in enumerate(tids):
            if tid in done:
                continue
            if vector_map is not None:
                spec = vector_map[tid]
                art = torch.load(spec["path"], weights_only=False)
                if "vectors" in art:
                    # per-layer raw vectors; alpha multiplies each layer's own scale
                    steerer_args = dict(vectors={int(L): v.float() for L, v in art["vectors"].items()},
                                        alpha=float(spec["alpha"]), token_slice=None)
                else:
                    vec = art["vector"].float().flatten()
                    steerer_args = dict(layer=int(art["layer"]), vector=vec / vec.norm().clamp_min(1e-12),
                                        alpha=float(spec["alpha"]), token_slice=None)
            if skill_map is not None:
                skill_prefix = build_skill_prompt(Path(skill_map[tid]).read_text())
            entry = manifest["tasks"][tid]["v0000"]
            gamefile = entry["gamefile"]
            eval_dataset = "eval_in_distribution" if "/valid_seen/" in gamefile else "eval_out_of_distribution"
            t0 = time.time()
            trace = []
            won = False
            try:
                env = build_alfworld_env(env_num=1, eval_dataset=eval_dataset, seed=42,
                                         is_train=False, specific_gamefiles=[gamefile])
                obs, infos = env.reset({})
                if args.temperature > 0.0:
                    torch.manual_seed(abs(hash((tid, args.sample_tag))) % (2 ** 31))
                for step_idx in range(args.max_steps):
                    user = skill_prefix + "\n" + obs["text"][0] if skill_prefix else obs["text"][0]
                    response = generate(model, tokenizer, args.device, user, steerer_args,
                                        args.max_new_tokens, temperature=args.temperature,
                                        gen_only=args.gen_only)
                    obs, rewards, dones, infos = env.step([response])
                    donef = bool(dones[0].item() if hasattr(dones[0], "item") else dones[0])
                    trace.append({
                        "step": step_idx,
                        "response": response,
                        "feedback": str(obs.get("anchor", [""])[0]),
                        "reward": float(rewards[0]),
                        "done": donef,
                    })
                    if donef:
                        won = bool(infos[0].get("won", False)) if infos else False
                        break
            except Exception as exc:  # noqa: BLE001
                trace.append({"error": str(exc)[:400]})
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            row = {
                "id": tid, "hard": int(won), "n_turns": len(trace),
                "category": ("repaired" if tid in cats["repaired"] else
                             "both_fail" if tid in cats["both_fail"] else
                             "both_success" if tid in cats["both_success"] else "broken"),
                "task_type": entry["task_type"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            fout.write(json.dumps(row) + "\n")
            fout.flush()
            (arm_dir / f"{tid.replace(':', '_')}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=1))
            print(json.dumps({"n": n, **row}), flush=True)

    rows = [json.loads(l) for l in results_path.read_text().splitlines() if l.strip()]
    summary = {
        "arm": args.arm_name, "skill": args.skill, "alpha": args.alpha,
        "layer": args.layer if not args.vector_path else steerer_args["layer"] if steerer_args else None,
        "vector_path": args.vector_path, "random_vector": args.random_vector,
        "episodes": len(rows), "successes": sum(r["hard"] for r in rows),
        "success_rate": sum(r["hard"] for r in rows) / max(1, len(rows)),
        "by_category": {},
    }
    for cat in set(r["category"] for r in rows):
        sub = [r for r in rows if r["category"] == cat]
        summary["by_category"][cat] = {"n": len(sub), "succ": sum(r["hard"] for r in sub)}
    (arm_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
