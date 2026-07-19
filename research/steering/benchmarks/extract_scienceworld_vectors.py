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

from research.steering.benchmarks.common import (
    HFSkillPolicy,
    collect_prompt_records,
    extract_prompt_vectors,
    load_results,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--bad-skill", required=True)
    p.add_argument("--good-skill", required=True)
    p.add_argument("--source-results", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--state-limit", type=int, default=120)
    p.add_argument("--layers", default="8,10,12,14,16,18,20,22,24,26,28")
    args = p.parse_args()

    source = load_results(args.source_results)
    prompts = collect_prompt_records(source, args.state_limit)
    layers = [int(x) for x in args.layers.split(",")]
    policy = HFSkillPolicy(str(Path(args.model_path).resolve()), args.device, 64)
    artifact = extract_prompt_vectors(
        policy,
        prompts,
        Path(args.bad_skill).read_text(),
        Path(args.good_skill).read_text(),
        layers,
    )
    artifact["sampling"] = "round_robin_episode_then_step"
    artifact["source_episodes"] = len(source)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, out / "skill_vectors.pt")
    printable = {k: v for k, v in artifact.items() if k != "vectors"}
    (out / "geometry.json").write_text(json.dumps(printable, indent=2))
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
