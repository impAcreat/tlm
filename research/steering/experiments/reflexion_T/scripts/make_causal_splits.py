#!/usr/bin/env python3
"""Freeze task-disjoint Dev-A/Dev-B splits from Train text-success tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()

    units = read_jsonl(args.units)
    pilot_tasks = sorted({
        row["task_id"] for row in units
        if row.get("split") == "dev" and row.get("text_success")
    })
    train_success = [
        row for row in units
        if row.get("split") == "train" and row.get("text_success")
    ]
    task_rows: dict[str, list[dict]] = defaultdict(list)
    for row in train_success:
        task_rows[row["task_id"]].append(row)
    by_type: dict[str, list[str]] = defaultdict(list)
    for task_id, rows in task_rows.items():
        task_types = {row["task_type"] for row in rows}
        if len(task_types) != 1:
            raise ValueError(f"inconsistent task types for {task_id}: {task_types}")
        by_type[next(iter(task_types))].append(task_id)

    total = len(task_rows)
    target_a = (total + 1) // 2
    base_a = sum(len(task_ids) // 2 for task_ids in by_type.values())
    odd_types = sorted(
        (task_type for task_type, task_ids in by_type.items() if len(task_ids) % 2),
        key=lambda value: stable_key(args.seed, value),
    )
    extra_a_types = set(odd_types[: target_a - base_a])

    split_a: list[str] = []
    split_b: list[str] = []
    for task_type in sorted(by_type):
        task_ids = sorted(by_type[task_type], key=lambda value: stable_key(args.seed, value))
        n_a = len(task_ids) // 2 + int(task_type in extra_a_types)
        split_a.extend(task_ids[:n_a])
        split_b.extend(task_ids[n_a:])

    def split_payload(task_ids: list[str]) -> dict:
        ids = sorted(task_ids)
        type_counts: dict[str, int] = defaultdict(int)
        paired_effective_tasks = 0
        for task_id in ids:
            rows = task_rows[task_id]
            type_counts[rows[0]["task_type"]] += 1
            paired_effective_tasks += int(any(row.get("paired_effective") for row in rows))
        return {
            "task_ids": ids,
            "n_tasks": len(ids),
            "task_type_counts": dict(sorted(type_counts.items())),
            "tasks_with_paired_effective": paired_effective_tasks,
        }

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "Train text_success tasks from frozen Qwen3-32B Reflexion collection",
        "seed": args.seed,
        "group_key": "task_id",
        "selection_rule": (
            "Within each task type, sort task_id by SHA256(seed|task_id), split as evenly as "
            "possible, and allocate odd remainders deterministically to reach 19/18."
        ),
        "pilot": {"task_ids": pilot_tasks, "n_tasks": len(pilot_tasks)},
        "eligible_train": {
            "n_text_success_records": len(train_success),
            "n_tasks": total,
        },
        "splits": {
            "Dev-A": split_payload(split_a),
            "Dev-B": split_payload(split_b),
        },
        "test_untouched": True,
    }
    if set(split_a) & set(split_b):
        raise AssertionError("Dev-A and Dev-B overlap")
    if (set(split_a) | set(split_b)) != set(task_rows):
        raise AssertionError("causal split does not cover all eligible Train tasks")
    if set(pilot_tasks) & (set(split_a) | set(split_b)):
        raise AssertionError("Pilot overlaps causal calibration splits")

    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
