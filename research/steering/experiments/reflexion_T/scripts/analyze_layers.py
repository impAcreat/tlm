#!/usr/bin/env python3
"""All-layer offline diagnostics; outputs candidates but never selects a final intervention layer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.steering.analysis.layer_selection import grouped_compiler_diagnostic


def load_records(paths: list[Path], split: str, subset: str) -> list[dict]:
    merged = {}
    for path in paths:
        shard = torch.load(path, weights_only=False)
        overlap = set(merged) & set(shard)
        if overlap:
            raise ValueError(f"duplicate units: {sorted(overlap)[:3]}")
        merged.update(shard)
    records = [record for record in merged.values() if record.get("split") == split]
    if subset != "all":
        records = [record for record in records if bool(record.get(subset))]
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset", choices=("all", "text_success", "paired_effective"), default="text_success")
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.inputs, args.split, args.subset)
    if len({record["task_id"] for record in records}) < 2:
        raise ValueError("need at least two task groups")
    vectors = np.stack([record["vector"].float().numpy() for record in records])
    text = np.stack([record["text_mean"].float().numpy() for record in records])
    groups = np.asarray([record["task_id"] for record in records])
    consistency = np.stack([record["consistency"].float().numpy() for record in records]).mean(0)
    residual_cos = grouped_compiler_diagnostic(text, vectors, groups, alpha=args.alpha, folds=args.folds)
    mean_vector = vectors.mean(0)
    norms = np.linalg.norm(vectors, axis=-1)
    residual_norms = np.linalg.norm(vectors - mean_vector[None], axis=-1)
    shared_ratio = np.linalg.norm(mean_vector, axis=-1) / (norms.mean(0) + 1e-12)
    unit_specific_ratio = residual_norms.mean(0) / (norms.mean(0) + 1e-12)
    rho = np.stack([record["natural_rho_median"].float().numpy() for record in records])

    layers = []
    for layer in range(vectors.shape[1]):
        layers.append({
            "layer": layer,
            "cross_state_consistency": float(consistency[layer]),
            "heldout_T_residual_cos": float(residual_cos[layer]),
            "mean_vector_norm": float(norms[:, layer].mean()),
            "shared_component_ratio": float(shared_ratio[layer]),
            "unit_specific_ratio": float(unit_specific_ratio[layer]),
            "natural_rho_median": float(np.median(rho[:, layer])),
        })
    # Diversified shortlist: top residual-cos layer inside each depth octile,
    # subject to above-median consistency. Final selection still requires Dev causality.
    consistency_floor = float(np.median(consistency))
    candidates = []
    for start in range(0, len(layers), 8):
        bucket = [row for row in layers[start:start + 8] if row["cross_state_consistency"] >= consistency_floor]
        if bucket:
            candidates.append(max(bucket, key=lambda row: row["heldout_T_residual_cos"])["layer"])
    result = {
        "protocol": {
            "split": args.split,
            "subset": args.subset,
            "group_key": "task_id",
            "n_units": len(records),
            "n_tasks": len(set(groups)),
            "final_layer_selected": False,
        },
        "offline_candidates": candidates,
        "layers": layers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"protocol": result["protocol"], "offline_candidates": candidates}, indent=2))


if __name__ == "__main__":
    main()
