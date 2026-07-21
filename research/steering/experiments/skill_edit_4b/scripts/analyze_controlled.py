"""Length-controlled skill-driven contrasts.

The naive full-trajectory inter/delta analysis is confounded by trajectory
length (failures are 50-step timeouts; n_steps alone predicts success at
~0.96 balanced accuracy). This script re-runs the skill-driven contrasts on
length-controlled representations:

  early5          mean of first <=5 step reps (all 32 layers)
  matched prefix  mean of first k=min(nA, nB, 10) step reps (8 stored layers)

and adds a skill-identifiability probe (v0000 vs step2 from early behavior,
grouped by task).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDS = ["v0000", "step1", "step2"]
STEP_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
NUM_LAYERS = 32
RNG = np.random.default_rng(42)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def mean_pairwise_cos(mat):
    if mat.shape[0] < 2:
        return float("nan")
    u = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sim = u @ u.T
    return float(sim[~np.eye(len(u), dtype=bool)].mean())


def grouped_probe(X, y, groups, folds=5):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    gkf = GroupKFold(n_splits=folds)
    accs = []
    for tr, te in gkf.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(accs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"
    manifest = json.loads((out_dir / "manifest.json").read_text())
    ids = manifest["ids"]

    data = {c: torch.load(out_dir / "reps" / f"{c}.pt", weights_only=False) for c in CONDS}
    E = {c: np.stack([data[c][t]["early5"].float().numpy() for t in ids]) for c in CONDS}
    Y = {c: np.array([data[c][t]["hard"] for t in ids]) for c in CONDS}

    cats = manifest["pairs"]["v0000_vs_step2"]
    r_idx = [ids.index(t) for t in cats["repaired"]]
    bs_idx = [ids.index(t) for t in cats["both_success"]]
    bf_idx = [ids.index(t) for t in cats["both_fail"]]

    out: dict = {"early5_delta": [], "matched_prefix_delta": [], "skill_probe": {}}

    # ---- early5 deltas across all 32 layers ----
    for L in range(NUM_LAYERS):
        D = E["step2"][:, L, :] - E["v0000"][:, L, :]
        X0 = E["v0000"][:, L, :]
        v_inter_early = unit(X0[Y["v0000"] == 1].mean(0) - X0[Y["v0000"] == 0].mean(0))
        perm = RNG.permutation(len(ids))
        Dnull = E["step2"][perm][: len(r_idx), L, :] - E["v0000"][r_idx, L, :]
        out["early5_delta"].append({
            "layer": L,
            "repaired_pairwise_cos": mean_pairwise_cos(D[r_idx]),
            "both_success_pairwise_cos": mean_pairwise_cos(D[bs_idx]),
            "both_fail_pairwise_cos": mean_pairwise_cos(D[bf_idx]),
            "all_pairwise_cos": mean_pairwise_cos(D),
            "null_pairwise_cos": mean_pairwise_cos(Dnull),
            "repaired_vs_early_inter_cos": float(np.dot(unit(D[r_idx].mean(0)), v_inter_early)),
            "all_vs_early_inter_cos": float(np.dot(unit(D.mean(0)), v_inter_early)),
            "repaired_mean_norm": float(np.linalg.norm(D[r_idx], axis=1).mean()),
        })

    # ---- matched-prefix deltas at stored step layers ----
    li = {L: i for i, L in enumerate(STEP_LAYERS)}
    prefix = {}
    for c in ("v0000", "step2"):
        prefix[c] = {}
        for t in ids:
            d = data[c][t]
            prefix[c][t] = d["step_reps"].float().numpy()  # [n, 8, 2560]
    for L in STEP_LAYERS:
        deltas = {}
        for t in ids:
            a, b = prefix["v0000"][t], prefix["step2"][t]
            k = min(len(a), len(b), 10)
            deltas[t] = b[:k, li[L]].mean(0) - a[:k, li[L]].mean(0)
        D = np.stack([deltas[t] for t in ids])
        perm = RNG.permutation(len(ids))
        Dnull = np.stack([
            prefix["step2"][ids[perm[i]]][: min(10, len(prefix["step2"][ids[perm[i]]])), li[L]].mean(0)
            - prefix["v0000"][ids[i]][: min(10, len(prefix["v0000"][ids[i]])), li[L]].mean(0)
            for i in range(len(r_idx))
        ])
        out["matched_prefix_delta"].append({
            "layer": L,
            "repaired_pairwise_cos": mean_pairwise_cos(D[r_idx]),
            "both_success_pairwise_cos": mean_pairwise_cos(D[bs_idx]),
            "both_fail_pairwise_cos": mean_pairwise_cos(D[bf_idx]),
            "null_pairwise_cos": mean_pairwise_cos(Dnull),
            "repaired_mean_norm": float(np.linalg.norm(D[r_idx], axis=1).mean()),
        })

    # ---- skill identifiability probe from early behavior (grouped by task) ----
    groups = np.concatenate([np.arange(len(ids)), np.arange(len(ids))])
    for L in (10, 14, 18, 22):
        X = np.concatenate([E["v0000"][:, L, :], E["step2"][:, L, :]])
        y = np.array([0] * len(ids) + [1] * len(ids))
        out["skill_probe"][f"L{L}_early5_v0000_vs_step2"] = grouped_probe(X, y, groups)

    (ana / "controlled_metrics.json").write_text(json.dumps(out, indent=2))

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ls = list(range(NUM_LAYERS))
    for key, color in (("repaired_pairwise_cos", "#2ca02c"), ("both_success_pairwise_cos", "#1f77b4"),
                       ("both_fail_pairwise_cos", "#d62728"), ("null_pairwise_cos", "#7f7f7f")):
        axes[0].plot(ls, [r[key] for r in out["early5_delta"]], color=color, label=key.replace("_pairwise_cos", ""))
        axes[1].plot(STEP_LAYERS, [r[key] for r in out["matched_prefix_delta"]], "o-", color=color,
                     label=key.replace("_pairwise_cos", ""))
    axes[0].set_title("early5 delta consistency (step2 - v0000)")
    axes[1].set_title("matched-prefix (k<=10) delta consistency")
    axes[2].plot(ls, [r["repaired_vs_early_inter_cos"] for r in out["early5_delta"]], label="repaired vs early-inter")
    axes[2].plot(ls, [r["all_vs_early_inter_cos"] for r in out["early5_delta"]], label="all vs early-inter")
    axes[2].set_title("alignment of early5 skill-delta with early success dir")
    for ax in axes:
        ax.set_xlabel("layer"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Length-controlled skill-driven contrasts (v0000 vs step2)")
    fig.tight_layout()
    fig.savefig(ana / "controlled_delta_sweep.png", dpi=150)
    plt.close(fig)

    best = max(out["early5_delta"], key=lambda r: r["repaired_pairwise_cos"])
    print("early5 best:", json.dumps(best, indent=1))
    print("matched prefix:", json.dumps(out["matched_prefix_delta"], indent=1))
    print("skill probe:", out["skill_probe"])


if __name__ == "__main__":
    main()
