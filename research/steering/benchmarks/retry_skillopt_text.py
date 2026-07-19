from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SKILLOPT = ROOT / "benchmarks/skillopt"
sys.path[:0] = [str(ROOT), str(SKILLOPT)]

from skillopt.gradient.aggregate import merge_patches
from skillopt.gradient.reflect import run_minibatch_reflect
from skillopt.optimizer.clip import rank_and_select
from skillopt.optimizer.skill import apply_patch_with_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Retry SkillOpt text stages from saved rollout trajectories.")
    parser.add_argument("--current-skill", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minibatch-size", type=int, default=3)
    parser.add_argument("--edit-budget", type=int, default=2)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    skill = Path(args.current_skill).read_text()
    results = [json.loads(line) for line in Path(args.results).read_text().splitlines() if line.strip()]
    raw = run_minibatch_reflect(
        results=results,
        skill_content=skill,
        prediction_dir=args.prediction_dir,
        patches_dir=str(out / "patches"),
        workers=1,
        failure_only=False,
        minibatch_size=args.minibatch_size,
        edit_budget=args.edit_budget,
        random_seed=42,
    )
    failure = [x["patch"] for x in raw if x and x.get("source_type", "failure") == "failure"]
    success = [x["patch"] for x in raw if x and x.get("source_type") == "success"]
    if not failure and not success:
        raise RuntimeError("SkillOpt reflection produced no usable patches")
    merged = merge_patches(skill, failure, success, batch_size=3, workers=1)
    selected = rank_and_select(skill, merged, max_edits=args.edit_budget)
    updated, report = apply_patch_with_report(skill, selected)
    (out / "skillopt_retry_skill.md").write_text(updated.strip() + "\n")
    (out / "text_stage_artifacts.json").write_text(
        json.dumps({"raw": raw, "merged": merged, "selected": selected, "apply_report": report}, indent=2)
    )
    print(f"wrote {out / 'skillopt_retry_skill.md'} ({len(skill)} -> {len(updated)} chars)")


if __name__ == "__main__":
    main()
