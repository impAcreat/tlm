from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
SKILLOPT = ROOT / "benchmarks/skillopt"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SKILLOPT))

from research.steering.benchmarks.common import random_matched_vectors
from research.steering.benchmarks.scienceworld import ScienceWorldAdapter, run_scienceworld


def _repeat_rate(rows: list[dict]) -> float:
    repeats = opportunities = 0
    for row in rows:
        actions = [step["action"] for step in row["trajectory"]]
        repeats += sum(a == b for a, b in zip(actions[1:], actions[:-1]))
        opportunities += max(0, len(actions) - 1)
    return repeats / max(1, opportunities)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split-dir", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--skill", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    p.add_argument("--eval-num", type=int, required=True)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--vector-path", default="")
    p.add_argument("--layers", default="")
    p.add_argument("--random-vector", action="store_true")
    p.add_argument(
        "--steer-mode",
        choices=["gen", "prefill_last", "prefill_last_gen", "all"],
        default="gen",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--steer-steps", type=int, default=0)
    p.add_argument("--stop-steer-score", type=float, default=101.0)
    p.add_argument("--knn-k", type=int, default=0)
    p.add_argument("--knn-temperature", type=float, default=0.1)
    p.add_argument("--shuffle-state-bank", action="store_true")
    p.add_argument("--online-good-skill", default="")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    adapter = ScienceWorldAdapter(
        args.split_dir, args.model_path, args.device, args.max_steps
    )
    adapter.setup({})
    if args.eval_split == "train":
        items = adapter.build_train_env(args.eval_num, args.seed)
    else:
        items = adapter.build_eval_env(args.eval_num, args.eval_split, args.seed)
    vectors = None
    state_bank = None
    if args.vector_path:
        artifact = torch.load(args.vector_path, map_location="cpu", weights_only=False)
        vectors = artifact.get("vectors", artifact)
        state_bank = artifact.get("state_bank")
        if args.layers:
            selected = {int(x) for x in args.layers.split(",")}
            vectors = {int(k): v for k, v in vectors.items() if int(k) in selected}
            if set(vectors) != selected:
                raise ValueError(f"missing requested layers: {sorted(selected - set(vectors))}")
        if args.random_vector:
            vectors = random_matched_vectors(vectors, seed=args.seed)
            state_bank = None
        elif args.shuffle_state_bank and state_bank:
            generator = torch.Generator().manual_seed(args.seed)
            state_bank = {
                layer: {**bank, "delta": bank["delta"][torch.randperm(len(bank["delta"]), generator=generator)]}
                for layer, bank in state_bank.items()
            }

    rows = run_scienceworld(
        items,
        Path(args.skill).read_text(),
        adapter.policy,
        args.max_steps,
        adapter.simplification,
        args.out_dir,
        vectors=vectors,
        alpha=args.alpha,
        steer_mode=args.steer_mode,
        steer_steps=args.steer_steps,
        stop_steer_score=args.stop_steer_score,
        state_bank=state_bank,
        knn_k=args.knn_k,
        knn_temperature=args.knn_temperature,
        online_good_skill=(Path(args.online_good_skill).read_text() if args.online_good_skill else None),
    )
    steps = [step for row in rows for step in row["trajectory"]]
    summary = {
        "n": len(rows),
        "ids": [row["id"] for row in rows],
        "hard": sum(row["hard"] for row in rows),
        "soft_mean": sum(row["soft"] for row in rows) / max(1, len(rows)),
        "scores": {row["id"]: row["score"] for row in rows},
        "mean_steps": sum(row["n_turns"] for row in rows) / max(1, len(rows)),
        "invalid_rate": sum(not step.get("is_valid", True) for step in steps) / max(1, len(steps)),
        "repeat_rate": _repeat_rate(rows),
        "eval_split": args.eval_split,
        "max_steps": args.max_steps,
        "steer_mode": args.steer_mode,
        "alpha": args.alpha,
        "random_vector": args.random_vector,
        "steer_steps": args.steer_steps,
        "stop_steer_score": args.stop_steer_score,
        "knn_k": args.knn_k,
        "knn_temperature": args.knn_temperature,
        "shuffle_state_bank": args.shuffle_state_bank,
        "online_good_skill": args.online_good_skill,
    }
    Path(args.out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
