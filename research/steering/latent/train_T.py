"""Train and evaluate the text->vector compiler T.

Data: t_dataset/unit_vectors_shard*.pt  (unit text reps + ground-truth
conditioning vectors, all layers).

Protocol (pre-declared):
  - Model: per-layer ridge regression  T_L : text_rep[L'] -> vector[L].
  - Primary metric: held-out cosine AFTER removing the mean unit vector
    (the shared "some instruction is present" component). A compiler that
    only learns the common component scores 0 here.
  - Baselines: (a) predict the training-mean vector; (b) label-shuffled T.
  - Splits: 5-fold on reflexion hints, grouped by source episode;
    cross-domain: train on all hints -> test on skillopt units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

LAYERS_OUT = (14, 18)
LAYERS_IN = (14, 18, 22)
RNG = np.random.default_rng(42)


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def cos_rows(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return (A * B).sum(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--text-rep", choices=("text_mean", "text_last"), default="text_mean")
    parser.add_argument("--alpha", type=float, default=100.0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    tdir = out_dir / "t_dataset"

    units = {}
    for f in sorted(tdir.glob("unit_vectors_shard*.pt")):
        units.update(torch.load(f, weights_only=False))
    recs = list(units.values())
    hints = [r for r in recs if r["source"] == "reflexion"]
    skedits = [r for r in recs if r["source"] == "skillopt"]
    print(f"{len(hints)} hints, {len(skedits)} skillopt units")

    results = {"n_hints": len(hints), "n_skillopt": len(skedits), "text_rep": args.text_rep,
               "consistency": {
                   "hints_cons_l18_mean": float(np.mean([r["cons_l18"] for r in hints])),
                   "skillopt_cons_l18_mean": float(np.mean([r["cons_l18"] for r in skedits])),
               }, "per_layer": {}}

    groups = np.array(["_".join(r["unit_id"].split("_")[:3]) for r in hints])  # episode group

    for L in LAYERS_OUT:
        Y = np.stack([r["vector"][L].float().numpy() for r in hints])
        Ysk = np.stack([r["vector"][L].float().numpy() for r in skedits])
        X = np.concatenate([np.stack([r[args.text_rep][Li].float().numpy() for r in hints])
                            for Li in LAYERS_IN], axis=1)
        Xsk = np.concatenate([np.stack([r[args.text_rep][Li].float().numpy() for r in skedits])
                              for Li in LAYERS_IN], axis=1)

        gkf = GroupKFold(n_splits=5)
        raw_cos, res_cos, base_cos = [], [], []
        for tr, te in gkf.split(X, groups=groups):
            model = Ridge(alpha=args.alpha)
            mu = Y[tr].mean(0)
            model.fit(X[tr], Y[tr])
            P = model.predict(X[te])
            raw_cos.append(cos_rows(P, Y[te]).mean())
            res_cos.append(cos_rows(P - mu, Y[te] - mu).mean())
            base_cos.append(cos_rows(np.tile(mu, (len(te), 1)), Y[te]).mean())
        # shuffled-label null for residual metric
        sh_cos = []
        for _ in range(3):
            perm = RNG.permutation(len(Y))
            for tr, te in gkf.split(X, groups=groups):
                m = Ridge(alpha=args.alpha)
                mu = Y[perm][tr].mean(0)
                m.fit(X[tr], Y[perm][tr])
                P = m.predict(X[te])
                sh_cos.append(cos_rows(P - mu, Y[perm][te] - mu).mean())
            break
        # cross-domain
        model = Ridge(alpha=args.alpha)
        mu = Y.mean(0)
        model.fit(X, Y)
        Psk = model.predict(Xsk)
        results["per_layer"][f"L{L}"] = {
            "heldout_raw_cos": float(np.mean(raw_cos)),
            "heldout_residual_cos": float(np.mean(res_cos)),
            "predict_mean_baseline_raw_cos": float(np.mean(base_cos)),
            "shuffled_null_residual_cos": float(np.mean(sh_cos)),
            "crossdomain_raw_cos": float(cos_rows(Psk, Ysk).mean()),
            "crossdomain_residual_cos": float(cos_rows(Psk - mu, Ysk - mu).mean()),
            "hint_vector_shared_frac": float(np.mean(cos_rows(Y, np.tile(Y.mean(0), (len(Y), 1))))),
        }

    (tdir / f"T_eval_{args.text_rep}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
