from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _load(path: str) -> dict[str, dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "results.jsonl"
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return {row["id"]: row for row in rows}


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round(q * (len(values) - 1))]


def _paired_stats(ref: dict[str, dict], arm: dict[str, dict], seed: int) -> dict:
    ids = sorted(set(ref) & set(arm))
    # Match the benchmark metric: ScienceWorld terminal failures can report
    # negative raw scores, while the adapter's soft metric clamps them to zero.
    deltas = [arm[i]["soft"] - ref[i]["soft"] for i in ids]
    rng = random.Random(seed)
    boot = [sum(rng.choice(deltas) for _ in deltas) / len(deltas) for _ in range(20000)]
    observed = abs(sum(deltas) / len(deltas))
    perm_extreme = 0
    for _ in range(50000):
        value = abs(sum(d * (-1 if rng.random() < 0.5 else 1) for d in deltas) / len(deltas))
        perm_extreme += value >= observed - 1e-12
    return {
        "n_paired": len(ids),
        "soft_delta_mean": sum(deltas) / len(deltas),
        "soft_delta_bootstrap_95ci": [_quantile(boot, 0.025), _quantile(boot, 0.975)],
        "paired_sign_flip_p": (perm_extreme + 1) / 50001,
        "improved": sum(d > 0 for d in deltas),
        "regressed": sum(d < 0 for d in deltas),
        "tied": sum(d == 0 for d in deltas),
        "per_task_delta": dict(zip(ids, deltas)),
    }


def _behavior(rows: dict[str, dict]) -> dict:
    steps = [step for row in rows.values() for step in row["trajectory"]]
    repeats = opportunities = 0
    for row in rows.values():
        actions = [step["action"] for step in row["trajectory"]]
        repeats += sum(a == b for a, b in zip(actions[1:], actions[:-1]))
        opportunities += max(0, len(actions) - 1)
    return {
        "invalid_rate": sum(not step.get("is_valid", True) for step in steps) / max(1, len(steps)),
        "repeat_rate": repeats / max(1, opportunities),
        "mean_steps": len(steps) / max(1, len(rows)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", required=True, help="name=run_dir")
    p.add_argument("--reference", default="bad")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260719)
    args = p.parse_args()
    arms = {name: _load(path) for name, path in (x.split("=", 1) for x in args.arms)}
    ref = arms[args.reference]
    summary = {"reference": args.reference, "arms": {}}
    for name, rows in arms.items():
        summary["arms"][name] = {
            "n": len(rows),
            "hard": sum(row["hard"] for row in rows.values()),
            "soft_mean": sum(row["soft"] for row in rows.values()) / len(rows),
            **_behavior(rows),
        }
        if name != args.reference:
            summary["arms"][name]["paired_vs_reference"] = _paired_stats(ref, rows, args.seed)
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
