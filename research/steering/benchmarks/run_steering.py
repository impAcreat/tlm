from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SKILLOPT = ROOT / "benchmarks/skillopt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SKILLOPT))

from research.steering.benchmarks.appworld import AppWorldAdapter, run_appworld
from research.steering.benchmarks.common import (
    collect_prompt_records, extract_prompt_vectors, load_results, random_matched_vectors,
)
from research.steering.benchmarks.scienceworld import ScienceWorldAdapter, run_scienceworld


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=["scienceworld", "appworld"], required=True)
    p.add_argument("--split-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--bad-skill", required=True)
    p.add_argument("--good-skill", required=True)
    p.add_argument("--source-results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--data-root", default="")
    p.add_argument("--state-limit", type=int, default=18)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--layers", default="14,18,22")
    p.add_argument("--steer-mode", choices=["gen", "prefill_last", "prefill_last_gen", "all"], default="gen")
    p.add_argument("--eval-split", choices=["val", "test"], default="test")
    p.add_argument("--eval-num", type=int, default=3)
    p.add_argument("--vector-path", default="")
    p.add_argument("--steer-steps", type=int, default=0)
    p.add_argument("--stop-steer-score", type=float, default=101.0)
    args = p.parse_args()

    bad = Path(args.bad_skill).read_text()
    good = Path(args.good_skill).read_text()
    source = load_results(args.source_results)
    prompts = collect_prompt_records(source, args.state_limit)
    if args.benchmark == "scienceworld":
        adapter = ScienceWorldAdapter(args.split_dir, args.model_path, args.device, args.max_steps)
    else:
        adapter = AppWorldAdapter(args.split_dir, args.model_path, args.data_root, args.device, args.max_steps)
    adapter.setup({})
    layers = [int(x) for x in args.layers.split(",")]
    if args.vector_path:
        artifact = torch.load(args.vector_path, map_location="cpu", weights_only=False)
        artifact["vectors"] = {
            int(k): v for k, v in artifact["vectors"].items() if int(k) in set(layers)
        }
    else:
        artifact = extract_prompt_vectors(adapter.policy, prompts, bad, good, layers)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out / "skill_vectors.pt")
    items = adapter.build_eval_env(args.eval_num, args.eval_split, 42)

    arms = {
        "bad": None,
        "good": None,
        "steered": artifact["vectors"],
        "random": random_matched_vectors(artifact["vectors"]),
    }
    summary = {
        "num_extraction_states": len(prompts),
        "alpha": args.alpha,
        "steer_mode": args.steer_mode,
        "eval_split": args.eval_split,
        "steer_steps": args.steer_steps,
        "stop_steer_score": args.stop_steer_score,
        "geometry": {k: v for k, v in artifact.items() if k != "vectors"},
        "arms": {},
    }
    rows_by_arm = {}
    for name, vectors in arms.items():
        skill = good if name == "good" else bad
        arm_dir = out / name
        if args.benchmark == "scienceworld":
            rows = run_scienceworld(
                items, skill, adapter.policy, args.max_steps, adapter.simplification,
                arm_dir,
                vectors=vectors,
                alpha=args.alpha,
                steer_mode=args.steer_mode,
                steer_steps=args.steer_steps,
                stop_steer_score=args.stop_steer_score,
            )
        else:
            rows = run_appworld(
                items, skill, adapter.policy, args.max_steps, adapter.data_root,
                arm_dir, vectors=vectors, alpha=args.alpha,
            )
        rows_by_arm[name] = rows
        summary["arms"][name] = {
            "n": len(rows), "hard": sum(x["hard"] for x in rows),
            "soft_mean": sum(x["soft"] for x in rows) / max(1, len(rows)),
            "mean_steps": sum(len(x["trajectory"]) for x in rows) / max(1, len(rows)),
        }
    bad_by_id = {row["id"]: row for row in rows_by_arm["bad"]}
    behavior = {}
    key = "action" if args.benchmark == "scienceworld" else "code"
    for name in ("good", "steered", "random"):
        changed = first_changed = 0
        for row in rows_by_arm[name]:
            bad = bad_by_id[row["id"]]
            seq = [step.get(key, "") for step in row["trajectory"]]
            bad_seq = [step.get(key, "") for step in bad["trajectory"]]
            changed += int(seq != bad_seq)
            first_changed += int(bool(seq and bad_seq and seq[0] != bad_seq[0]))
        n = max(1, len(rows_by_arm[name]))
        behavior[name] = {
            "trajectory_change_rate_vs_bad": changed / n,
            "first_decision_change_rate_vs_bad": first_changed / n,
        }
    summary["behavior_vs_bad"] = behavior
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
