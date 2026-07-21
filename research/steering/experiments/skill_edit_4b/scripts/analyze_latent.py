"""Latent-space analysis for SkillOpt selection rollouts.

Part A  NPM-style inter: success vs failure separation per condition, per layer.
Part B  Skill-driven: task-paired deltas across skill conditions.
Part C  NPM-style intra: effective vs degenerate steps inside failed v0000 rollouts.

Outputs metrics.json and figures under <out-dir>/analysis/.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CONDS = ["v0000", "step1", "step2"]
STEP_LAYERS = [2, 6, 10, 14, 18, 22, 26, 30]
NUM_LAYERS = 32
RNG = np.random.default_rng(42)

COND_LABEL = {"v0000": "rough_v1 (41.4%)", "step1": "step-1 cand (52.1%)", "step2": "step-2 cand (62.1%)"}


def probe_acc(X: np.ndarray, y: np.ndarray, folds: int = 5) -> float:
    if len(set(y)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=skf, scoring="balanced_accuracy").mean())


def centroid_cos(a: np.ndarray, b: np.ndarray) -> float:
    ca, cb = a.mean(0), b.mean(0)
    return float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def mean_pairwise_cos(mat: np.ndarray) -> float:
    if mat.shape[0] < 2:
        return float("nan")
    u = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    sim = u @ u.T
    mask = ~np.eye(len(u), dtype=bool)
    return float(sim[mask].mean())


def scatter2d(ax, emb, labels, colors, names, title):
    for val, color, name in zip(sorted(set(labels)), colors, names):
        pts = emb[np.array(labels) == val]
        ax.scatter(pts[:, 0], pts[:, 1], s=14, alpha=0.75, c=color, label=f"{name} (n={len(pts)})")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    ana = out_dir / "analysis"
    ana.mkdir(exist_ok=True)
    manifest = json.loads((out_dir / "manifest.json").read_text())
    ids = manifest["ids"]

    data = {c: torch.load(out_dir / "reps" / f"{c}.pt", weights_only=False) for c in CONDS}
    # traj_all / early5 arrays: [n_tasks, 32, 2560]
    T = {c: np.stack([data[c][t]["traj_all"].float().numpy() for t in ids]) for c in CONDS}
    E = {c: np.stack([data[c][t]["early5"].float().numpy() for t in ids]) for c in CONDS}
    Y = {c: np.array([data[c][t]["hard"] for t in ids]) for c in CONDS}
    NSTEP = {c: np.array([data[c][t]["n_steps"] for t in ids]) for c in CONDS}
    TTYPE = np.array([manifest["tasks"][t]["v0000"]["task_type"] for t in ids])

    metrics: dict = {"inter": {}, "length_baseline": {}, "early5": {}, "pairs": {}, "intra": {}, "transfer": {}}

    # ---------- Part A: inter ----------
    for c in CONDS:
        rows = []
        for L in range(NUM_LAYERS):
            X = T[c][:, L, :]
            pos, neg = X[Y[c] == 1], X[Y[c] == 0]
            rows.append({
                "layer": L,
                "probe_bacc": probe_acc(X, Y[c]),
                "centroid_cos": centroid_cos(pos, neg),
                "silhouette": float(silhouette_score(X, Y[c], metric="cosine")),
            })
        metrics["inter"][c] = rows
        metrics["length_baseline"][c] = probe_acc(NSTEP[c].reshape(-1, 1).astype(float), Y[c])
        metrics["early5"][c] = [
            {"layer": L, "probe_bacc": probe_acc(E[c][:, L, :], Y[c])} for L in range(NUM_LAYERS)
        ]

    # layer sweep figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for c in CONDS:
        ls = [r["layer"] for r in metrics["inter"][c]]
        axes[0].plot(ls, [r["probe_bacc"] for r in metrics["inter"][c]], label=COND_LABEL[c])
        axes[1].plot(ls, [r["centroid_cos"] for r in metrics["inter"][c]], label=COND_LABEL[c])
        axes[2].plot(ls, [r["probe_bacc"] for r in metrics["early5"][c]], label=COND_LABEL[c])
        axes[0].axhline(metrics["length_baseline"][c], ls="--", lw=0.8, alpha=0.5)
    axes[0].set_title("full-traj probe balanced acc (dashed: n_steps baseline)")
    axes[1].set_title("success/fail centroid cosine")
    axes[2].set_title("early-5-step probe balanced acc")
    for ax in axes:
        ax.set_xlabel("layer")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Inter (success vs fail) separation across layers — behavior-only trajectory reps")
    fig.tight_layout()
    fig.savefig(ana / "inter_layer_sweep.png", dpi=150)
    plt.close(fig)

    best_layer = {
        c: int(max(metrics["inter"][c], key=lambda r: r["probe_bacc"])["layer"]) for c in CONDS
    }
    metrics["best_layer"] = best_layer

    # PCA / t-SNE per condition at its best layer
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for j, c in enumerate(CONDS):
        L = best_layer[c]
        X = T[c][:, L, :]
        p = PCA(n_components=2).fit_transform(X - X.mean(0))
        scatter2d(axes[0, j], p, Y[c], ["#d62728", "#2ca02c"], ["fail", "success"],
                  f"{COND_LABEL[c]} PCA L{L}")
        ts = TSNE(n_components=2, perplexity=20, random_state=42, init="pca").fit_transform(X)
        scatter2d(axes[1, j], ts, Y[c], ["#d62728", "#2ca02c"], ["fail", "success"],
                  f"{COND_LABEL[c]} t-SNE L{L}")
    fig.suptitle("Success vs failure clusters per skill condition (valid_seen 140)")
    fig.tight_layout()
    fig.savefig(ana / "inter_pca_tsne.png", dpi=150)
    plt.close(fig)

    # cross-condition transfer of the success probe at a shared layer
    Lstar = int(np.argmax(np.mean(
        [[r["probe_bacc"] for r in metrics["inter"][c]] for c in CONDS], axis=0)))
    metrics["shared_best_layer"] = Lstar
    for ctrain in CONDS:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        clf.fit(T[ctrain][:, Lstar, :], Y[ctrain])
        for ctest in CONDS:
            if ctest == ctrain:
                continue
            from sklearn.metrics import balanced_accuracy_score
            pred = clf.predict(T[ctest][:, Lstar, :])
            metrics["transfer"][f"{ctrain}->{ctest}"] = float(balanced_accuracy_score(Y[ctest], pred))

    # combined map: all conditions at shared best layer
    Xall = np.concatenate([T[c][:, Lstar, :] for c in CONDS])
    cond_lab = sum([[c] * len(ids) for c in CONDS], [])
    out_lab = np.concatenate([Y[c] for c in CONDS])
    ts = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(Xall)
    pc = PCA(n_components=2).fit_transform(Xall - Xall.mean(0))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    palette = {"v0000": "#1f77b4", "step1": "#ff7f0e", "step2": "#2ca02c"}
    for ax, emb, name in ((axes[0], pc, "PCA"), (axes[1], ts, "t-SNE")):
        for c in CONDS:
            m = np.array([cl == c for cl in cond_lab])
            for hv, marker in ((1, "o"), (0, "x")):
                pts = emb[m & (out_lab == hv)]
                ax.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.7, c=palette[c], marker=marker,
                           label=f"{c} {'succ' if hv else 'fail'}")
        ax.set_title(f"{name} @ layer {Lstar}")
        ax.legend(fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("All conditions, colored by skill, marker by outcome")
    fig.tight_layout()
    fig.savefig(ana / "combined_condition_map.png", dpi=150)
    plt.close(fig)

    # ---------- Part B: paired skill deltas ----------
    for other in ("step1", "step2"):
        cats = manifest["pairs"][f"v0000_vs_{other}"]
        rows = []
        for L in range(NUM_LAYERS):
            D = T[other][:, L, :] - T["v0000"][:, L, :]
            entry = {"layer": L}
            for cat, tids in cats.items():
                idx = [ids.index(t) for t in tids]
                if len(idx) >= 2:
                    entry[f"{cat}_pairwise_cos"] = mean_pairwise_cos(D[idx])
                    entry[f"{cat}_mean_norm"] = float(np.linalg.norm(D[idx], axis=1).mean())
            # shuffled-pair null for repaired
            r_idx = [ids.index(t) for t in cats["repaired"]]
            if len(r_idx) >= 2:
                perm = RNG.permutation(len(ids))
                Dnull = T[other][perm][: len(r_idx), L, :] - T["v0000"][r_idx, L, :]
                entry["repaired_null_pairwise_cos"] = mean_pairwise_cos(Dnull)
                # alignment with inter direction of v0000 at this layer
                X0 = T["v0000"][:, L, :]
                v_inter = unit(X0[Y["v0000"] == 1].mean(0) - X0[Y["v0000"] == 0].mean(0))
                v_rep = unit(D[r_idx].mean(0))
                entry["repaired_vs_inter_cos"] = float(np.dot(v_rep, v_inter))
            rows.append(entry)
        metrics["pairs"][f"v0000_vs_{other}"] = rows

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for other, style in (("step1", "--"), ("step2", "-")):
        rows = metrics["pairs"][f"v0000_vs_{other}"]
        ls = [r["layer"] for r in rows]
        for cat, color in (("repaired", "#2ca02c"), ("both_fail", "#d62728"),
                           ("both_success", "#1f77b4"), ("repaired_null", "#7f7f7f")):
            key = f"{cat}_pairwise_cos"
            vals = [r.get(key, np.nan) for r in rows]
            axes[0].plot(ls, vals, style, color=color, lw=1.2,
                         label=f"{other}:{cat}" if other == "step2" or cat == "repaired" else None)
        axes[1].plot(ls, [r.get("repaired_vs_inter_cos", np.nan) for r in rows], style,
                     label=f"v0000->{other}")
        axes[2].plot(ls, [r.get("repaired_mean_norm", np.nan) for r in rows], style, color="#2ca02c",
                     label=f"{other} repaired")
        axes[2].plot(ls, [r.get("both_success_mean_norm", np.nan) for r in rows], style, color="#1f77b4",
                     label=f"{other} both_success")
    axes[0].set_title("paired-delta mean pairwise cosine by category")
    axes[1].set_title("cos(mean repaired delta, inter success-fail dir)")
    axes[2].set_title("delta norms")
    for ax in axes:
        ax.set_xlabel("layer"); ax.grid(alpha=0.3); ax.legend(fontsize=6)
    fig.suptitle("Skill-driven paired deltas (same task, bad vs good skill)")
    fig.tight_layout()
    fig.savefig(ana / "pair_delta_sweep.png", dpi=150)
    plt.close(fig)

    # delta scatter at shared best layer: PCA of deltas colored by category
    D = T["step2"][:, Lstar, :] - T["v0000"][:, Lstar, :]
    cats = manifest["pairs"]["v0000_vs_step2"]
    cat_of = {}
    for cat, tids in cats.items():
        for t in tids:
            cat_of[t] = cat
    labels = [cat_of[t] for t in ids]
    p = PCA(n_components=2).fit_transform(D - D.mean(0))
    fig, ax = plt.subplots(figsize=(7, 6))
    for cat, color in (("repaired", "#2ca02c"), ("broken", "#9467bd"),
                       ("both_success", "#1f77b4"), ("both_fail", "#d62728")):
        pts = p[np.array(labels) == cat]
        ax.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.75, c=color, label=f"{cat} (n={len(pts)})")
    ax.legend(); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"PCA of per-task deltas (step2 - v0000) @ layer {Lstar}")
    fig.tight_layout()
    fig.savefig(ana / "pair_delta_pca.png", dpi=150)
    plt.close(fig)

    # ---------- Part C: intra on failed v0000 ----------
    li = {L: i for i, L in enumerate(STEP_LAYERS)}
    intra_rows = []
    pos_all, neg_all = {L: [] for L in STEP_LAYERS}, {L: [] for L in STEP_LAYERS}
    for t in ids:
        d = data["v0000"][t]
        if d["hard"] == 1:
            continue
        flags = d["flags"]
        sr = d["step_reps"].float().numpy()  # [n, 8, 2560]
        for s in range(d["n_steps"]):
            degenerate = flags["invalid"][s] or flags["repeat"][s]
            for L in STEP_LAYERS:
                (neg_all if degenerate else pos_all)[L].append(sr[s, li[L]])
    for L in STEP_LAYERS:
        P, N = np.stack(pos_all[L]), np.stack(neg_all[L])
        Xs = np.concatenate([P, N]); ys = np.array([1] * len(P) + [0] * len(N))
        intra_rows.append({
            "layer": L, "n_pos": len(P), "n_neg": len(N),
            "centroid_cos": centroid_cos(P, N),
            "probe_bacc": probe_acc(Xs, ys, folds=3),
        })
    metrics["intra"]["v0000_failed"] = intra_rows

    Lbest_intra = max(intra_rows, key=lambda r: r["probe_bacc"])["layer"]
    P, N = np.stack(pos_all[Lbest_intra]), np.stack(neg_all[Lbest_intra])
    sub = min(800, len(P), len(N))
    Pi = P[RNG.choice(len(P), sub, replace=False)]
    Ni = N[RNG.choice(len(N), sub, replace=False)]
    Xs = np.concatenate([Pi, Ni])
    ts = TSNE(n_components=2, perplexity=30, random_state=42, init="pca").fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter2d(ax, ts, [1] * sub + [0] * sub, ["#d62728", "#2ca02c"],
              ["degenerate step", "effective step"],
              f"intra steps t-SNE @ layer {Lbest_intra} (failed v0000 trajs)")
    fig.tight_layout()
    fig.savefig(ana / "intra_step_tsne.png", dpi=150)
    plt.close(fig)

    (ana / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # console summary
    print("best layers per cond:", best_layer, "shared:", Lstar)
    for c in CONDS:
        r = metrics["inter"][c][Lstar]
        print(f"{c} @L{Lstar}: probe {r['probe_bacc']:.3f} centroid_cos {r['centroid_cos']:.3f} "
              f"len_baseline {metrics['length_baseline'][c]:.3f} "
              f"early5 {metrics['early5'][c][Lstar]['probe_bacc']:.3f}")
    print("transfer:", metrics["transfer"])
    rows = metrics["pairs"]["v0000_vs_step2"]
    rbest = max(rows, key=lambda r: r.get("repaired_pairwise_cos", -9))
    print("repaired delta consistency best:", rbest)
    print("intra:", intra_rows)


if __name__ == "__main__":
    main()
