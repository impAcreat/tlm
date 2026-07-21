#!/usr/bin/env python3
"""Freeze task-disjoint Train/Dev/Test task-seed groups before generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def task_type(manifest: dict, task_id: str) -> str:
    return str(manifest["tasks"][task_id]["v0000"]["task_type"])


def old_failure_ids(manifest: dict, prefix: str) -> list[str]:
    return [
        task_id
        for task_id in manifest["ids"]
        if task_id.startswith(prefix) and not manifest["tasks"][task_id]["v0000"]["hard"]
    ]


def split_seen_tasks(manifest: dict, seed: int, dev_fraction: float):
    buckets: dict[str, list[str]] = defaultdict(list)
    for task_id in old_failure_ids(manifest, "val:"):
        buckets[task_type(manifest, task_id)].append(task_id)
    train, dev = [], []
    for kind, task_ids in sorted(buckets.items()):
        random.Random(stable_seed(seed, kind)).shuffle(task_ids)
        n_dev = max(1, round(len(task_ids) * dev_fraction))
        dev.extend(task_ids[:n_dev])
        train.extend(task_ids[n_dev:])
    return sorted(train), sorted(dev)


def expand(manifest: dict, split: str, task_ids: list[str], seeds: int) -> list[dict]:
    groups = []
    for task_id in task_ids:
        for task_seed in range(seeds):
            groups.append(
                {
                    "group_id": f"{split}_{task_id.replace(':', '')}_s{task_seed:02d}",
                    "split": split,
                    "task_id": task_id,
                    "task_seed": task_seed,
                    "task_type": task_type(manifest, task_id),
                }
            )
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--train-seeds", type=int, default=4)
    parser.add_argument("--dev-seeds", type=int, default=2)
    parser.add_argument("--test-seeds", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    train_tasks, dev_tasks = split_seen_tasks(manifest, args.seed, args.dev_fraction)
    test_tasks = sorted(old_failure_ids(manifest, "test:"))
    assert not (set(train_tasks) & set(dev_tasks))
    assert not ((set(train_tasks) | set(dev_tasks)) & set(test_tasks))

    groups = (
        expand(manifest, "train", train_tasks, args.train_seeds)
        + expand(manifest, "dev", dev_tasks, args.dev_seeds)
        + expand(manifest, "test", test_tasks, args.test_seeds)
    )
    random.Random(args.seed).shuffle(groups)
    serialized = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode()
    plan = {
        "version": 1,
        "seed": args.seed,
        "selection": {
            "train_dev": "valid_seen tasks failed by frozen 4B v0000 baseline",
            "test": "valid_unseen tasks failed by frozen 4B v0000 baseline",
            "task_disjoint": True,
            "dev_fraction": args.dev_fraction,
        },
        "task_counts": {
            "train": len(train_tasks),
            "dev": len(dev_tasks),
            "test": len(test_tasks),
        },
        "group_counts": {
            split: sum(group["split"] == split for group in groups)
            for split in ("train", "dev", "test")
        },
        "split_hash": hashlib.sha256(serialized).hexdigest(),
        "groups": groups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({k: plan[k] for k in ("task_counts", "group_counts", "split_hash")}, indent=2))


if __name__ == "__main__":
    main()
