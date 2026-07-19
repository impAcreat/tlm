"""Consolidate candidate steering vectors and cross-analyze them.

Families:
  v_prompt_good_minus_bad[L]   prompt-conditioned skill contrast (same state)
  v_prompt_good_minus_none[L]
  v_prompt_bad_minus_none[L]
  v_behav_early5_all[L]        mean early5 behavioral delta over all 140 pairs
  v_behav_early5_repaired[L]
  v_inter_full[L]              success-fail full-traj direction (length-confounded)

Outputs: vectors/*.pt (unit vectors, load_steering_vector-compatible),
analysis/vector_alignment.png, analysis/vector_metrics.json.
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

CONDS = ["v0000", "step2"]
NUM_LAYERS = 32


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def mean_pairwise_cos(mat, max_n=400, seed=0):
    if mat.shape[0] < 2:
        return float("nan")
    if mat.shape[0] > max_n:
        idx = np.random.default_rng(seed).choice(mat.shape[0], max_n, replace=False)
        mat = mat[idx]
    u = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sim = u @ u.T
    return float(sim[~np.eye(len(u), dtype=bool)].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"
    vec_dir = out_dir / "vectors"
    vec_dir.mkdir(exist_ok=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    ids = manifest["ids"]

    pd = torch.load(out_dir / "prompt_deltas.pt", weights_only=False)
    states = pd["states"]
    D = {k: v.float().numpy() for k, v in pd["deltas"].items()}  # [n, 32, 2560]

    data = {c: torch.load(out_dir / "reps" / f"{c}.pt", weights_only=False) for c in CONDS}
    E = {c: np.stack([data[c][t]["early5"].float().numpy() for t in ids]) for c in CONDS}
    T0 = np.stack([data["v0000"][t]["traj_all"].float().numpy() for t in ids])
    Y0 = np.array([data["v0000"][t]["hard"] for t in ids])
    r_idx = [ids.index(t) for t in manifest["pairs"]["v0000_vs_step2"]["repaired"]]

    Eall = E["step2"] - E["v0000"]  # [140, 32, 2560]

    families = {}
    metrics = {"prompt_consistency": [], "alignments": {}}

    for L in range(NUM_LAYERS):
        families.setdefault("v_prompt_good_minus_bad", []).append(unit(D["good_minus_bad"][:, L].mean(0)))
        families.setdefault("v_prompt_good_minus_none", []).append(unit(D["good_minus_none"][:, L].mean(0)))
        families.setdefault("v_prompt_bad_minus_none", []).append(unit(D["bad_minus_none"][:, L].mean(0)))
        families.setdefault("v_behav_early5_all", []).append(unit(Eall[:, L].mean(0)))
        families.setdefault("v_behav_early5_repaired", []).append(unit(Eall[r_idx, L].mean(0)))
        X = T0[:, L]
        families.setdefault("v_inter_full", []).append(unit(X[Y0 == 1].mean(0) - X[Y0 == 0].mean(0)))
        metrics["prompt_consistency"].append({
            "layer": L,
            "gmb_pairwise_cos": mean_pairwise_cos(D["good_minus_bad"][:, L]),
            "gmn_pairwise_cos": mean_pairwise_cos(D["good_minus_none"][:, L]),
            "bmn_pairwise_cos": mean_pairwise_cos(D["bad_minus_none"][:, L]),
            "gmb_mean_norm": float(np.linalg.norm(D["good_minus_bad"][:, L], axis=1).mean()),
            "gmb_norm_of_mean": float(np.linalg.norm(D["good_minus_bad"][:, L].mean(0))),
        })

    families = {k: np.stack(v) for k, v in families.items()}  # [32, 2560]

    # pairwise alignment across families per layer
    names = list(families)
    align = np.zeros((len(names), len(names), NUM_LAYERS))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            for L in range(NUM_LAYERS):
                align[i, j, L] = float(np.dot(families[a][L], families[b][L]))
    metrics["alignments"] = {
        f"{a}|{b}": [float(align[i, j, L]) for L in range(NUM_LAYERS)]
        for i, a in enumerate(names) for j, b in enumerate(names) if i < j
    }

    # additivity check: good-bad vs (good-none - bad-none)
    addv = []
    for L in range(NUM_LAYERS):
        lhs = unit(D["good_minus_bad"][:, L].mean(0))
        rhs = unit(D["good_minus_none"][:, L].mean(0) - D["bad_minus_none"][:, L].mean(0))
        addv.append(float(np.dot(lhs, rhs)))
    metrics["additivity_gmb_vs_gmn_minus_bmn"] = addv

    # per-category / per-step consistency of good_minus_bad at its best layer
    cons = [r["gmb_pairwise_cos"] for r in metrics["prompt_consistency"]]
    Lbest = int(np.argmax(cons))
    metrics["gmb_best_layer"] = Lbest
    cat_arr = np.array([s["category"] for s in states])
    step_arr = np.array([s["step"] for s in states])
    metrics["gmb_by_category"] = {
        c: mean_pairwise_cos(D["good_minus_bad"][cat_arr == c, Lbest]) for c in sorted(set(cat_arr))
    }
    metrics["gmb_by_step"] = {
        int(s): mean_pairwise_cos(D["good_minus_bad"][step_arr == s, Lbest]) for s in sorted(set(step_arr))
    }

    # save vectors for steering (raw mean, not unit, plus unit) at all layers
    for fam, mat in families.items():
        torch.save({"vectors_unit": torch.tensor(mat), "num_layers": NUM_LAYERS, "family": fam},
                   vec_dir / f"{fam}_all_layers.pt")
    # per-layer single-vector files compatible with load_steering_vector
    for fam in ("v_prompt_good_minus_bad", "v_prompt_good_minus_none", "v_behav_early5_all", "v_inter_full"):
        for L in (10, 14, 18, 22, Lbest):
            torch.save({"vector": torch.tensor(families[fam][L]), "layer": int(L), "family": fam},
                       vec_dir / f"{fam}_l{L}.pt")
    # raw (unnormalized) prompt delta means for alpha calibration
    torch.save({"mean_delta": torch.tensor(np.stack([D["good_minus_bad"][:, L].mean(0) for L in range(NUM_LAYERS)])),
                "mean_norm_per_layer": [r["gmb_mean_norm"] for r in metrics["prompt_consistency"]]},
               vec_dir / "gmb_raw_means.pt")

    (ana / "vector_metrics.json").write_text(json.dumps(metrics, indent=2))

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ls = list(range(NUM_LAYERS))
    axes[0].plot(ls, [r["gmb_pairwise_cos"] for r in metrics["prompt_consistency"]], label="good-bad")
    axes[0].plot(ls, [r["gmn_pairwise_cos"] for r in metrics["prompt_consistency"]], label="good-none")
    axes[0].plot(ls, [r["bmn_pairwise_cos"] for r in metrics["prompt_consistency"]], label="bad-none")
    axes[0].set_title("prompt-delta cross-state consistency")
    for pairname in ("v_prompt_good_minus_bad|v_behav_early5_all",
                     "v_prompt_good_minus_bad|v_inter_full",
                     "v_behav_early5_all|v_inter_full",
                     "v_prompt_good_minus_bad|v_prompt_good_minus_none"):
        axes[1].plot(ls, metrics["alignments"][pairname], label=pairname.replace("v_prompt_", "p:").replace("v_behav_", "b:").replace("v_inter_full", "inter"))
    axes[1].set_title("cross-family alignment (cosine per layer)")
    axes[2].plot(ls, [r["gmb_mean_norm"] for r in metrics["prompt_consistency"]], label="mean ||delta||")
    axes[2].plot(ls, [r["gmb_norm_of_mean"] for r in metrics["prompt_consistency"]], label="||mean delta||")
    axes[2].set_title("good-bad prompt delta norms")
    for ax in axes:
        ax.set_xlabel("layer"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Steering-vector families: consistency, alignment, scale")
    fig.tight_layout()
    fig.savefig(ana / "vector_alignment.png", dpi=150)
    plt.close(fig)

    print("gmb best layer:", Lbest)
    print("gmb consistency around best:", cons[max(0, Lbest - 2): Lbest + 3])
    print("by category:", metrics["gmb_by_category"])
    print("by step:", metrics["gmb_by_step"])
    print("alignment p:good-bad vs b:early5_all @Lbest:",
          metrics["alignments"]["v_prompt_good_minus_bad|v_behav_early5_all"][Lbest])
    print("alignment p:good-bad vs inter @Lbest:",
          metrics["alignments"]["v_prompt_good_minus_bad|v_inter_full"][Lbest])
    print("additivity @Lbest:", addv[Lbest])


if __name__ == "__main__":
    main()
