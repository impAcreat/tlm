#!/usr/bin/env python3
"""Fit and cross-validate the ridge text-to-steering compiler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from research.steering.core.compiler.ridge import RidgeCompiler
from research.steering.core.metrics.geometry import cosine_rows


def load_records(paths: list[Path]) -> list[dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        shard = torch.load(path, weights_only=False)
        overlap = set(merged).intersection(shard)
        if overlap:
            raise ValueError(f"duplicate unit ids: {sorted(overlap)[:3]}")
        merged.update(shard)
    return list(merged.values())


def arrays(records: list[dict], text_layers: list[int], target_layer: int, group_key: str):
    x = np.concatenate(
        [np.stack([r["text_mean"][layer].float().numpy() for r in records]) for layer in text_layers],
        axis=1,
    )
    y = np.stack([r["vector"][target_layer].float().numpy() for r in records])
    groups = np.asarray([r[group_key] for r in records])
    return x.astype(np.float32), y.astype(np.float32), groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-layers", type=int, nargs="+", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--subset", choices=("all", "text_success", "paired_effective"), default="text_success")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--group-key", default="task_id")
    parser.add_argument("--split", choices=("train", "dev", "all"), default="train")
    args = parser.parse_args()

    records = load_records(args.inputs)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if args.subset != "all":
        records = [record for record in records if bool(record.get(args.subset))]
    x, y, groups = arrays(records, args.text_layers, args.target_layer, args.group_key)
    n_splits = min(args.folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("compiler evaluation needs at least two episode groups")

    predictions = np.zeros_like(y)
    train_means = np.zeros_like(y)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    for fold_id, (train, test) in enumerate(GroupKFold(n_splits).split(x, groups=groups)):
        compiler = RidgeCompiler(alpha=args.alpha).fit(x[train], y[train])
        predictions[test] = compiler.predict(x[test])
        train_means[test] = y[train].mean(0)
        fold_ids[test] = fold_id

    raw_cos = cosine_rows(predictions, y)
    residual_cos = cosine_rows(predictions - train_means, y - train_means)
    final = RidgeCompiler(alpha=args.alpha).fit(x, y)
    artifact = {
        "compiler": final,
        "protocol": {
            "method": "ridge",
            "alpha": args.alpha,
            "subset": args.subset,
            "text_layers": args.text_layers,
            "target_layer": args.target_layer,
            "group_key": args.group_key,
            "n_splits": n_splits,
        },
        "metrics": {
            "n": len(records),
            "heldout_raw_cos": float(raw_cos.mean()),
            "heldout_residual_cos": float(residual_cos.mean()),
        },
        "per_unit": [
            {
                "unit_id": record["unit_id"],
                "episode_id": record["episode_id"],
                "task_id": record.get("task_id"),
                "fold": int(fold_ids[i]),
                "raw_cos": float(raw_cos[i]),
                "residual_cos": float(residual_cos[i]),
            }
            for i, record in enumerate(records)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in artifact.items() if k != "compiler"}, indent=2) + "\n"
    )
    print(json.dumps(artifact["metrics"], indent=2))


if __name__ == "__main__":
    main()
