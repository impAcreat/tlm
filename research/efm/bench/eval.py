"""Bench-agnostic results aggregator + arm comparison for EFM idea-validation.

Reads any run dir that follows the common output contract:
  - results.jsonl        : one JSON per episode, with at least {hard, soft}
  - feedback_state.json  : optional; episodes[].trace[].step_feedback (efm arms)

Works for ANY bench (ALFWorld / Terminal-Bench / GIAI2) as long as it emits
this contract, so there is no per-bench eval code.

Usage:
  python -m research.efm.bench.eval <run_dir> [<run_dir> ...] [--labels raw efm]
  python -m research.efm.bench.eval <run_dir> --progress   # monitor a live run
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

try:
    from research.efm.quality import score_feedback
except Exception:  # pragma: no cover - eval still works for task-only arms
    score_feedback = None


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def load_run(run_dir: str) -> dict:
    d = Path(run_dir)
    results = _read_jsonl(d / "results.jsonl")
    state_path = d / "feedback_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
    return {"dir": str(d), "name": d.name, "results": results, "state": state}


def _ci95(values: list[float]) -> tuple[float, float]:
    """Bootstrap 95% CI of the mean."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(0)
    means = []
    n = len(values)
    for _ in range(1000):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return (means[25], means[975])


def summarize(run: dict) -> dict:
    results = run["results"]
    hard = [float(r.get("hard", 0) or 0) for r in results]
    soft = [float(r.get("soft", 0) or 0) for r in results]
    out = {"name": run["name"], "n": len(results)}
    if hard:
        lo, hi = _ci95(hard)
        out["hard"] = sum(hard) / len(hard)
        out["hard_ci"] = (lo, hi)
        out["soft"] = sum(soft) / len(soft)
    # feedback-quality + interaction metrics (efm arms only)
    state = run["state"]
    if state and score_feedback is not None:
        steps = fb = c = p = e = 0
        ep_steps = []
        for ep in state.get("episodes", []):
            trace = ep.get("trace", [])
            ep_steps.append(len(trace))
            for i, row in enumerate(trace):
                steps += 1
                sf = row["step_feedback"]
                if sf.get("fallback"):
                    fb += 1
                q = score_feedback(sf, action=row["action"], raw_observation=row["raw_observation"],
                                   recent_actions=[t["action"] for t in trace[:i]])
                c += q.consistency; p += q.completeness; e += q.efficiency
        if steps:
            out.update({"steps": steps, "fallback_pct": 100 * fb / steps,
                        "consistency_pct": 100 * c / steps, "completeness_pct": 100 * p / steps,
                        "efficiency_pct": 100 * e / steps,
                        "avg_steps": sum(ep_steps) / len(ep_steps) if ep_steps else 0})
    return out


def _fmt(s: dict) -> str:
    base = f"{s['name'][:42]:<42} n={s['n']:<4}"
    if "hard" in s:
        lo, hi = s["hard_ci"]
        base += f" hard={s['hard']:.3f}[{lo:.2f},{hi:.2f}] soft={s['soft']:.3f}"
    if "completeness_pct" in s:
        base += (f" | cons={s['consistency_pct']:.0f} comp={s['completeness_pct']:.0f}"
                 f" eff={s['efficiency_pct']:.0f} fb={s['fallback_pct']:.0f}% steps={s['avg_steps']:.1f}")
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--progress", action="store_true", help="just print n + running scores")
    args = ap.parse_args()
    summaries = []
    for i, rd in enumerate(args.run_dirs):
        run = load_run(rd)
        s = summarize(run)
        if args.labels and i < len(args.labels):
            s["name"] = f"{args.labels[i]}:{s['name']}"
        summaries.append(s)
    print(f"{'arm':<42} {'metrics'}")
    for s in summaries:
        print(_fmt(s))
    # pairwise task-score delta when >=2 arms
    if len(summaries) >= 2 and all("hard" in s for s in summaries[:2]):
        a, b = summaries[0], summaries[1]
        print(f"\nΔhard ({b['name'].split(':')[0]} - {a['name'].split(':')[0]}) = {b['hard']-a['hard']:+.3f}"
              f"  Δsoft = {b['soft']-a['soft']:+.3f}")


if __name__ == "__main__":
    main()
