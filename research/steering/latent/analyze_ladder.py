"""Round 4 analysis: step-ladder geometry, unit-edit geometry, and
skill-text-semantics -> conditioning-vector alignment.

Inputs (same replayed states, aligned by construction):
  prompt_deltas.pt        reps_last_token: bad(v0000), good(=s2), none   [1213, 32, d]
  prompt_deltas_step3.pt  reps_last_token: good(=s3)                     [1213, 32, d]
  ladder_reps.pt          reps: s1, s1_e{0..k}  on 600-state subsample; text_reps
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

NUM_LAYERS = 32


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def mpc(mat, max_n=400, seed=0):
    if mat.shape[0] < 2:
        return float("nan")
    if mat.shape[0] > max_n:
        mat = mat[np.random.default_rng(seed).choice(mat.shape[0], max_n, replace=False)]
    u = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sim = u @ u.T
    return float(sim[~np.eye(len(u), dtype=bool)].mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"

    p2 = torch.load(out_dir / "prompt_deltas.pt", weights_only=False)
    p3 = torch.load(out_dir / "prompt_deltas_step3.pt", weights_only=False)
    lad = torch.load(out_dir / "ladder_reps.pt", weights_only=False)
    idx = np.array(lad["subsample_indices"])

    H = {
        "v0000": p2["reps_last_token"]["bad"][idx].float().numpy(),
        "s2": p2["reps_last_token"]["good"][idx].float().numpy(),
        "none": p2["reps_last_token"]["none"][idx].float().numpy(),
        "s3": p3["reps_last_token"]["good"][idx].float().numpy(),
        "s1": lad["reps"]["s1"].float().numpy(),
    }
    unit_names = [k for k in lad["reps"] if k.startswith("s1_e")]
    for k in unit_names:
        H[k] = lad["reps"][k].float().numpy()

    out: dict = {"ladder": [], "units": [], "semantics": {}}

    # ---------- A. step ladder ----------
    chains = [("v0000", "s1", "d01"), ("s1", "s2", "d12"), ("s2", "s3", "d23"), ("v0000", "s2", "d02")]
    D = {}
    for a, b, name in chains:
        D[name] = H[b] - H[a]  # [n, 32, d]
    for L in range(NUM_LAYERS):
        row = {"layer": L}
        for name in ("d01", "d12", "d23", "d02"):
            row[f"{name}_consistency"] = mpc(D[name][:, L])
            row[f"{name}_norm"] = float(np.linalg.norm(D[name][:, L].mean(0)))
        for x, y in (("d01", "d12"), ("d12", "d23"), ("d01", "d23")):
            row[f"cos_{x}_{y}"] = float(np.dot(unit(D[x][:, L].mean(0)), unit(D[y][:, L].mean(0))))
        out["ladder"].append(row)

    # ---------- B. unit edits ----------
    V = {k: H[k] - H["s1"] for k in unit_names}  # per-state unit deltas
    for L in range(NUM_LAYERS):
        row = {"layer": L}
        means = {}
        for k in unit_names:
            means[k] = V[k][:, L].mean(0)
            row[f"{k}_consistency"] = mpc(V[k][:, L])
            row[f"{k}_norm"] = float(np.linalg.norm(means[k]))
        for i, a in enumerate(unit_names):
            for b in unit_names[i + 1:]:
                row[f"cos_{a}_{b}"] = float(np.dot(unit(means[a]), unit(means[b])))
        ssum = sum(means.values())
        row["cos_sum_vs_d12"] = float(np.dot(unit(ssum), unit(D["d12"][:, L].mean(0))))
        row["norm_ratio_sum_vs_d12"] = float(
            np.linalg.norm(ssum) / (np.linalg.norm(D["d12"][:, L].mean(0)) + 1e-12))
        out["units"].append(row)

    # ---------- C. semantics -> vector alignment ----------
    T = {k: v.float().numpy() for k, v in lad["text_reps"].items()}
    pairs = [("v0000", "s1", "d01"), ("s1", "s2", "d12"), ("s2", "s3", "d23"), ("v0000", "s2", "d02")]
    per_layer = []
    for L in range(NUM_LAYERS):
        row = {"layer": L}
        # matched alignment: text-diff vs conditioning delta of the SAME pair
        M = np.zeros((len(pairs), len(pairs)))
        for i, (a, b, dn) in enumerate(pairs):
            tdiff = unit(T[b][L] - T[a][L])
            for j, (_, _, dn2) in enumerate(pairs):
                M[i, j] = float(np.dot(tdiff, unit(D[dn2][:, L].mean(0))))
        row["diag_mean"] = float(np.mean(np.diag(M)))
        row["offdiag_mean"] = float(np.mean(M[~np.eye(len(pairs), dtype=bool)]))
        row["matrix"] = M.round(4).tolist()
        # unit-level: edit content rep vs unit conditioning vector
        ucos = []
        for i, k in enumerate(unit_names):
            td = unit(T[k][L] - T["s1"][L])
            ucos.append(float(np.dot(td, unit(V[k][:, L].mean(0)))))
        row["unit_textdiff_vs_vec"] = ucos
        per_layer.append(row)
    out["semantics"]["per_layer"] = per_layer

    (ana / "ladder_metrics.json").write_text(json.dumps(out, indent=2))

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    ls = list(range(NUM_LAYERS))
    for name, c in (("d01", "#1f77b4"), ("d12", "#ff7f0e"), ("d23", "#2ca02c")):
        axes[0].plot(ls, [r[f"{name}_consistency"] for r in out["ladder"]], color=c, label=f"{name} consistency")
        axes[0].plot(ls, [r[f"{name}_norm"] / 10 for r in out["ladder"]], "--", color=c, alpha=0.5,
                     label=f"{name} norm/10")
    axes[0].set_title("ladder increments: consistency + norm")
    for pairn, c in (("cos_d01_d12", "#1f77b4"), ("cos_d12_d23", "#ff7f0e"), ("cos_d01_d23", "#2ca02c")):
        axes[1].plot(ls, [r[pairn] for r in out["ladder"]], color=c, label=pairn)
    axes[1].set_title("are increments the same direction?")
    axes[2].plot(ls, [r["diag_mean"] for r in per_layer], label="text-diff vs matched delta (diag)")
    axes[2].plot(ls, [r["offdiag_mean"] for r in per_layer], label="mismatched (offdiag)")
    axes[2].plot(ls, [float(np.mean(r["unit_textdiff_vs_vec"])) for r in per_layer],
                 label="unit edits: text-diff vs vec")
    axes[2].set_title("semantics -> vector alignment")
    for ax in axes:
        ax.set_xlabel("layer"); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Step-ladder and unit-edit geometry; text-semantics alignment")
    fig.tight_layout()
    fig.savefig(ana / "ladder_geometry.png", dpi=150)

    mid = [r for r in out["ladder"] if 10 <= r["layer"] <= 22]
    print("=== ladder (mid-layer means) ===")
    for k in ("d01_consistency", "d12_consistency", "d23_consistency",
              "cos_d01_d12", "cos_d12_d23", "cos_d01_d23",
              "d01_norm", "d12_norm", "d23_norm"):
        print(" ", k, round(float(np.mean([r[k] for r in mid])), 4))
    midu = [r for r in out["units"] if 10 <= r["layer"] <= 22]
    print("=== units (mid-layer means) ===")
    for k in sorted(midu[0]):
        if k != "layer":
            print(" ", k, round(float(np.mean([r[k] for r in midu])), 4))
    mids = [r for r in per_layer if 10 <= r["layer"] <= 22]
    print("=== semantics (mid-layer means) ===")
    print("  diag", round(float(np.mean([r["diag_mean"] for r in mids])), 4),
          "offdiag", round(float(np.mean([r["offdiag_mean"] for r in mids])), 4),
          "units", round(float(np.mean([np.mean(r["unit_textdiff_vs_vec"]) for r in mids])), 4))


if __name__ == "__main__":
    main()
