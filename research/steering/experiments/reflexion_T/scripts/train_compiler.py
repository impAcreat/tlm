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


def select_alpha(x, y, groups, alphas, folds):
    """Select ridge regularization using only grouped inner folds."""
    unique_groups = np.unique(groups)
    n_splits = min(folds, len(unique_groups))
    if n_splits < 2:
        raise ValueError("alpha selection needs at least two task groups")
    splits = list(GroupKFold(n_splits).split(x, groups=groups))
    scores = {}
    for alpha in alphas:
        values = []
        for train, test in splits:
            mean = y[train].mean(0)
            pred = RidgeCompiler(alpha=float(alpha)).fit(x[train], y[train]).predict(x[test])
            values.extend(cosine_rows(pred - mean, y[test] - mean))
        scores[float(alpha)] = float(np.mean(values))
    selected = max(sorted(scores), key=lambda alpha: scores[alpha])
    return selected, scores


def nested_grouped_predictions(x, y, groups, *, alphas, outer_folds, inner_folds, seed=42):
    """Generate leakage-safe outer predictions plus mean and shuffled controls."""
    n_splits = min(outer_folds, len(np.unique(groups)))
    if n_splits < 2:
        raise ValueError("compiler evaluation needs at least two task groups")
    predictions = np.zeros_like(y)
    shuffled_predictions = np.zeros_like(y)
    train_means = np.zeros_like(y)
    fold_ids = np.full(len(y), -1, dtype=np.int64)
    selected_alphas = []
    inner_scores = []
    for fold_id, (train, test) in enumerate(GroupKFold(n_splits).split(x, groups=groups)):
        alpha, scores = select_alpha(x[train], y[train], groups[train], alphas, inner_folds)
        predictions[test] = RidgeCompiler(alpha=alpha).fit(x[train], y[train]).predict(x[test])
        rng = np.random.default_rng(seed + fold_id)
        shuffled_y = y[train][rng.permutation(len(train))]
        shuffled_predictions[test] = RidgeCompiler(alpha=alpha).fit(
            x[train], shuffled_y
        ).predict(x[test])
        train_means[test] = y[train].mean(0)
        fold_ids[test] = fold_id
        selected_alphas.append(alpha)
        inner_scores.append(scores)
    return {
        "predictions": predictions,
        "shuffled_predictions": shuffled_predictions,
        "train_means": train_means,
        "fold_ids": fold_ids,
        "selected_alphas": selected_alphas,
        "inner_scores": inner_scores,
        "n_splits": n_splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text-layers", type=int, nargs="+", required=True)
    parser.add_argument("--target-layer", type=int, required=True)
    parser.add_argument("--subset", choices=("all", "text_success", "paired_effective"), default="text_success")
    parser.add_argument("--alpha", type=float, help="fixed-alpha compatibility mode")
    parser.add_argument("--alphas", type=float, nargs="+", default=[1.0, 10.0, 100.0, 1000.0, 10000.0])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    parser.add_argument("--group-key", default="task_id")
    parser.add_argument("--split", choices=("train", "dev", "all"), default="train")
    args = parser.parse_args()

    records = load_records(args.inputs)
    if args.split != "all":
        records = [record for record in records if record.get("split") == args.split]
    if args.subset != "all":
        records = [record for record in records if bool(record.get(args.subset))]
    x, y, groups = arrays(records, args.text_layers, args.target_layer, args.group_key)
    alpha_grid = [args.alpha] if args.alpha is not None else args.alphas
    nested = nested_grouped_predictions(
        x, y, groups, alphas=alpha_grid, outer_folds=args.folds,
        inner_folds=args.inner_folds,
    )
    predictions = nested["predictions"]
    shuffled_predictions = nested["shuffled_predictions"]
    train_means = nested["train_means"]
    fold_ids = nested["fold_ids"]

    raw_cos = cosine_rows(predictions, y)
    residual_cos = cosine_rows(predictions - train_means, y - train_means)
    shuffled_raw_cos = cosine_rows(shuffled_predictions, y)
    shuffled_residual_cos = cosine_rows(shuffled_predictions - train_means, y - train_means)
    mean_raw_cos = cosine_rows(train_means, y)
    final_alpha, final_alpha_scores = select_alpha(x, y, groups, alpha_grid, args.inner_folds)
    final = RidgeCompiler(alpha=final_alpha).fit(x, y)
    artifact = {
        "compiler": final,
        "protocol": {
            "method": "ridge",
            "alpha_grid": [float(value) for value in alpha_grid],
            "final_alpha": final_alpha,
            "outer_selected_alphas": nested["selected_alphas"],
            "outer_inner_scores": nested["inner_scores"],
            "final_alpha_scores": final_alpha_scores,
            "subset": args.subset,
            "text_layers": args.text_layers,
            "target_layer": args.target_layer,
            "group_key": args.group_key,
            "n_splits": nested["n_splits"],
            "inner_folds": args.inner_folds,
        },
        "metrics": {
            "n": len(records),
            "heldout_raw_cos": float(raw_cos.mean()),
            "heldout_residual_cos": float(residual_cos.mean()),
            "mean_predictor_raw_cos": float(mean_raw_cos.mean()),
            "mean_predictor_residual_cos": 0.0,
            "shuffled_label_raw_cos": float(shuffled_raw_cos.mean()),
            "shuffled_label_residual_cos": float(shuffled_residual_cos.mean()),
            "residual_cos_gain_over_shuffled": float(
                residual_cos.mean() - shuffled_residual_cos.mean()
            ),
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
