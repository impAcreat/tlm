"""Graded behavioral effect of steering: do steered rollouts move toward
good-skill behavior even when success does not flip?

For each arm and episode:
  divergence_step   first step where the arm's action differs from pilot_bad_base
  jaccard_good      action-set Jaccard overlap with the step2 (good-skill) vllm rollout
  jaccard_bad_vllm  action-set Jaccard with the v0000 (bad-skill) vllm rollout
  repeat_rate       fraction of steps repeating an earlier action
  invalid_rate      fraction of steps whose feedback contains "Nothing happens"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ACTION_RE = re.compile(r"<action>(.*?)</action>", re.DOTALL | re.IGNORECASE)


def actions_from_arm(trace: list[dict]) -> list[str]:
    acts = []
    for step in trace:
        if "response" not in step:
            continue
        m = ACTION_RE.search(step["response"] or "")
        acts.append(m.group(1).strip().lower() if m else "")
    return acts


def actions_from_vllm(conv: list[dict]) -> list[str]:
    return [str(s.get("action") or "").strip().lower() for s in conv]


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / max(1, len(a | b))


def rates(trace: list[dict], acts: list[str]) -> tuple[float, float]:
    seen = set()
    rep = 0
    inv = 0
    n = 0
    for step, act in zip([s for s in trace if "response" in s], acts):
        n += 1
        if act and act in seen:
            rep += 1
        if act:
            seen.add(act)
        if "nothing happens" in str(step.get("feedback") or "").lower():
            inv += 1
    return (rep / max(1, n), inv / max(1, n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--arms", required=True, help="comma list; first is the reference bad baseline")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    arms = args.arms.split(",")
    ref = arms[0]

    ref_actions = {}
    for f in (out_dir / "steered_eval" / ref).glob("val_*.json"):
        tid = f.stem.replace("val_", "val:")
        ref_actions[tid] = actions_from_arm(json.loads(f.read_text()))

    summary = {}
    for arm in arms:
        rows = []
        for f in sorted((out_dir / "steered_eval" / arm).glob("val_*.json")):
            tid = f.stem.replace("val_", "val:")
            trace = json.loads(f.read_text())
            acts = actions_from_arm(trace)
            entry = manifest["tasks"][tid]
            good_acts = actions_from_vllm(json.loads(Path(entry["step2"]["conversation"]).read_text()))
            bad_acts = actions_from_vllm(json.loads(Path(entry["v0000"]["conversation"]).read_text()))
            div = float("nan")
            if arm != ref and tid in ref_actions:
                div = next((i for i, (x, y) in enumerate(zip(acts, ref_actions[tid])) if x != y),
                           min(len(acts), len(ref_actions[tid])))
            rep, inv = rates(trace, acts)
            rows.append({
                "id": tid,
                "divergence_step": div,
                "jaccard_good": jaccard(set(a for a in acts if a), set(a for a in good_acts if a)),
                "jaccard_bad_vllm": jaccard(set(a for a in acts if a), set(a for a in bad_acts if a)),
                "repeat_rate": rep,
                "invalid_rate": inv,
                "n_steps": len(acts),
            })
        if rows:
            keys = ["jaccard_good", "jaccard_bad_vllm", "repeat_rate", "invalid_rate"]
            agg = {k: sum(r[k] for r in rows if r[k] == r[k]) / max(1, sum(1 for r in rows if r[k] == r[k])) for k in keys}
            agg["mean_divergence_step"] = (
                sum(r["divergence_step"] for r in rows if r["divergence_step"] == r["divergence_step"])
                / max(1, sum(1 for r in rows if r["divergence_step"] == r["divergence_step"]))
            )
            agg["episodes"] = len(rows)
            summary[arm] = {"aggregate": agg, "rows": rows}

    (out_dir / "analysis" / "behav_effect.json").write_text(json.dumps(summary, indent=2))
    for arm, s in summary.items():
        print(arm, json.dumps(s["aggregate"]))


if __name__ == "__main__":
    main()
