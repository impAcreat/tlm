"""Prepare the causal validation of compiler T.

Selects the top-K step1-sourced hints by extracted-vector consistency,
retrains T (ridge, text_mean, L14+18+22 -> L18) EXCLUDING the eval episodes,
and writes per-task vector files plus vector-map JSONs for four arms:
  extracted   ground-truth unit vector (upper bound)
  predicted   T output, unit-normalized (held-out)
  random      per-task random unit vector
Alpha per task = 3.9 x extracted norm @L18 for every injected arm (dose
matched; direction is the only variable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge

LAYERS_IN = (14, 18, 22)
L_OUT = 18
K = 20


def unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    tdir = out_dir / "t_dataset"
    vdir = tdir / "causal_vectors"
    vdir.mkdir(exist_ok=True)

    units = {}
    for f in sorted(tdir.glob("unit_vectors_shard*.pt")):
        units.update(torch.load(f, weights_only=False))
    hints = {k: r for k, r in units.items() if r["source"] == "reflexion"}

    # unit_id format: hint_{cond}_{valXXXX}_{n}
    def ep_key(r):
        return r["unit_id"].rsplit("_", 1)[0]

    def task_of(r):
        raw = r["unit_id"].split("_")[2]  # e.g. val0001
        return f"val:{raw[3:]}"

    # eval selection: step1-sourced, best hint per episode, top-K by consistency
    step1 = [r for r in hints.values() if r["unit_id"].startswith("hint_step1_")]
    by_ep: dict[str, list] = {}
    for r in step1:
        by_ep.setdefault(ep_key(r), []).append(r)
    best = [max(v, key=lambda r: r["cons_l18"]) for v in by_ep.values()]
    best.sort(key=lambda r: -r["cons_l18"])
    eval_recs = best[:K]
    eval_eps = {ep_key(r) for r in eval_recs}
    eval_tasks = {task_of(r) for r in eval_recs}
    assert len(eval_tasks) == len(eval_recs), "one task per record expected"

    # train T on all hints from non-eval episodes
    train = [r for r in hints.values() if ep_key(r) not in eval_eps]
    X = np.concatenate([np.stack([r["text_mean"][L].float().numpy() for r in train])
                        for L in LAYERS_IN], axis=1)
    Y = np.stack([r["vector"][L_OUT].float().numpy() for r in train])
    model = Ridge(alpha=100.0)
    model.fit(X, Y)
    print(f"T trained on {len(train)} hints (excluded {len(eval_eps)} eval episodes)")

    maps = {"extracted": {}, "predicted": {}, "random": {}}
    meta = []
    g = torch.Generator().manual_seed(11)
    for r in eval_recs:
        tid = task_of(r)
        ex = r["vector"][L_OUT].float().numpy()
        alpha = float(3.9 * np.linalg.norm(ex))
        x = np.concatenate([r["text_mean"][L].float().numpy() for L in LAYERS_IN])[None]
        pred = model.predict(x)[0]
        rnd = torch.randn(len(ex), generator=g).numpy()
        for arm, vec in (("extracted", unit(ex)), ("predicted", unit(pred)), ("random", unit(rnd))):
            p = vdir / f"{arm}_{tid.replace(':', '_')}.pt"
            torch.save({"vector": torch.tensor(vec, dtype=torch.float32), "layer": L_OUT,
                        "family": f"T_causal_{arm}", "unit_id": r["unit_id"]}, p)
            maps[arm][tid] = {"path": str(p), "alpha": alpha}
        meta.append({"task_id": tid, "unit_id": r["unit_id"], "cons_l18": r["cons_l18"],
                     "extracted_norm": float(np.linalg.norm(ex)),
                     "predicted_norm": float(np.linalg.norm(pred)),
                     "cos_pred_extracted": float(np.dot(unit(pred), unit(ex))),
                     "alpha": alpha, "hint": str(r.get("text", ""))[:160]})
    for arm, m in maps.items():
        (vdir / f"map_{arm}.json").write_text(json.dumps(m, indent=1))
    (vdir / "eval_meta.json").write_text(json.dumps(meta, indent=1))
    cs = [m["cos_pred_extracted"] for m in meta]
    print(f"eval tasks: {sorted(t for t in maps['extracted'])}")
    print(f"cos(predicted, extracted): mean {np.mean(cs):.3f} min {min(cs):.3f} max {max(cs):.3f}")
    print(f"alpha range: {min(m['alpha'] for m in meta):.2f} - {max(m['alpha'] for m in meta):.2f}")


if __name__ == "__main__":
    main()
