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
    artifact = extract_prompt_vectors(adapter.policy, prompts, bad, good, [14, 18, 22])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out / "skill_vectors.pt")
    items = adapter.build_eval_env(3, "test", 42)

    arms = {"bad": None, "good": "good", "steered": artifact["vectors"], "random": random_matched_vectors(artifact["vectors"])}
    summary = {"num_extraction_states": len(prompts), "geometry": {k: v for k, v in artifact.items() if k != "vectors"}, "arms": {}}
    for name, vectors in arms.items():
        skill = good if name == "good" else bad
        arm_dir = out / name
        if args.benchmark == "scienceworld":
            rows = run_scienceworld(items, skill, adapter.policy, args.max_steps, adapter.simplification, arm_dir, vectors=vectors)
        else:
            rows = run_appworld(items, skill, adapter.policy, args.max_steps, adapter.data_root, arm_dir, vectors=vectors)
        summary["arms"][name] = {
            "n": len(rows), "hard": sum(x["hard"] for x in rows),
            "soft_mean": sum(x["soft"] for x in rows) / max(1, len(rows)),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

