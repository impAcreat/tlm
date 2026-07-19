from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def find_project_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "benchmarks" / "skillopt").exists():
            return parent
    raise RuntimeError(f"could not find project root from {start}")


ROOT = find_project_root(Path(__file__).resolve())
SKILLOPT_ROOT = ROOT / "benchmarks" / "skillopt"
for path in (str(ROOT), str(SKILLOPT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from research.steering.rollouts.rollout_hf import (  # noqa: E402
    RolloutStep,
    SteeringSpec,
    build_messages,
    extract_action,
    generate_response,
    load_steering_vector,
    normalize_model_response,
    summarize_steps,
)
from skillopt.envs.alfworld.rollout import _get_task_type, build_alfworld_env  # noqa: E402


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_cases(results_jsonl: str | Path, *, max_episodes: int, offset: int = 0) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(results_jsonl) if row.get("gamefile")]
    if offset:
        rows = rows[offset:]
    cases = rows[:max_episodes]
    if not cases:
        raise ValueError(f"no gamefiles found in {results_jsonl}")
    return cases


def load_skill(path: str) -> str:
    if not path:
        return ""
    skill_path = Path(path)
    if not skill_path.exists() or skill_path.is_dir():
        return ""
    return skill_path.read_text(encoding="utf-8")


def load_model(model_path: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    model.eval()
    return model, tokenizer


def _scalar_bool(value: Any) -> bool:
    try:
        if hasattr(value, "item"):
            return bool(value.item())
        if isinstance(value, (list, tuple)) and value:
            return bool(value[0])
        return bool(value)
    except Exception:
        return False


def _task_description(initial_obs: dict[str, Any]) -> str:
    anchor = str((initial_obs.get("anchor") or [""])[0])
    marker = "Your task is to: "
    idx = anchor.find(marker)
    return anchor[idx + len(marker):].strip() if idx >= 0 else ""


def run_episode(
    *,
    model,
    tokenizer,
    case: dict[str, Any],
    mode: str,
    steering: SteeringSpec | None,
    skill_content: str,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    seed: int,
    enable_thinking: bool | None,
) -> dict[str, Any]:
    gamefile = str(case["gamefile"])
    eval_dataset = "eval_out_of_distribution"
    if "/valid_seen/" in gamefile:
        eval_dataset = "eval_in_distribution"
    elif "/train/" in gamefile:
        eval_dataset = "train"

    env_manager = build_alfworld_env(
        env_num=1,
        eval_dataset=eval_dataset,
        seed=seed,
        is_train=eval_dataset == "train",
        specific_gamefiles=[gamefile],
    )
    steps: list[RolloutStep] = []
    raw_records: list[dict[str, Any]] = []
    won = False
    done = False
    infos: list[dict[str, Any]] = []
    try:
        obs, infos = env_manager.reset({})
        task_description = _task_description(obs) or str(case.get("task_description") or "")
        for step_idx in range(max_steps):
            prompt = obs["text"][0]
            messages = build_messages(prompt, skill_content=skill_content)
            raw_response = generate_response(
                model,
                tokenizer,
                messages,
                device=device,
                steering=steering,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                enable_thinking=enable_thinking,
            )
            response = normalize_model_response(raw_response)
            action = extract_action(response)
            obs, rewards, dones, infos = env_manager.step([response])
            reward = float(rewards[0])
            done = _scalar_bool(dones[0])
            valid = _scalar_bool(infos[0].get("is_action_valid")) if infos else False
            post_obs = str(obs.get("anchor", [""])[0])
            step = RolloutStep(
                step=step_idx,
                action=action,
                response=response,
                observation=post_obs,
                reward=reward,
                done=done,
                valid=valid,
                prompt=prompt,
            )
            steps.append(step)
            raw_records.append(
                {
                    **step.to_dict(),
                    "raw_model_response": raw_response,
                    "gamefile": gamefile,
                    "mode": mode,
                }
            )
            if done:
                won = bool(infos[0].get("won", False)) if infos else False
                break
        else:
            won = False
    finally:
        close = getattr(env_manager, "close", None)
        if callable(close):
            close()

    metrics = summarize_steps(steps)
    fail_reason = "" if won else ("Timeout after %d steps" % max_steps if not done else "Episode ended without completing the task")
    return {
        "id": str(case.get("id") or Path(gamefile).parent.name),
        "mode": mode,
        "hard": 1 if won else 0,
        "soft": 1.0 if won else 0.0,
        "fail_reason": fail_reason,
        "gamefile": gamefile,
        "task_type": str(case.get("task_type") or _get_task_type(gamefile)),
        "task_description": task_description,
        **metrics,
        "trace": raw_records,
    }


def run_suite(
    *,
    model,
    tokenizer,
    cases: list[dict[str, Any]],
    mode: str,
    steering: SteeringSpec | None,
    skill_content: str,
    max_steps: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
    seed: int,
    enable_thinking: bool | None,
    out_dir: Path,
) -> list[dict[str, Any]]:
    mode_dir = out_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases):
        episode = run_episode(
            model=model,
            tokenizer=tokenizer,
            case=case,
            mode=mode,
            steering=steering,
            skill_content=skill_content,
            max_steps=max_steps,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=device,
            seed=seed + idx,
            enable_thinking=enable_thinking,
        )
        trace = episode.pop("trace")
        task_id = episode["id"].replace(":", "_").replace("/", "_")
        episode_dir = mode_dir / task_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        (episode_dir / "conversation.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        results.append(episode)
        print(json.dumps({"mode": mode, "episode": idx, **episode}, ensure_ascii=False), flush=True)
    with (mode_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return results


def summarize_suite(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    return {
        "episodes": total,
        "successes": sum(int(r["hard"]) for r in results),
        "success_rate": (sum(int(r["hard"]) for r in results) / total) if total else 0.0,
        "mean_turns": (sum(float(r["n_turns"]) for r in results) / total) if total else 0.0,
        "invalid_actions": sum(int(r["invalid_actions"]) for r in results),
        "repeated_actions": sum(int(r["repeated_actions"]) for r in results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run true HF activation steering on real ALFWorld rollout.")
    parser.add_argument("--model-path", default="models/Qwen3-4B-Instruct-2507")
    parser.add_argument("--vector-path", default="research/steering/runs/alfworld_qwen3_4b_constfix_pca_l16_neutral_split70_20260706/steering_vector.pt")
    parser.add_argument("--results-jsonl", default="benchmarks/skillopt/outputs/skillopt4b_promptv4_initial_raw_test32/results.jsonl")
    parser.add_argument("--skill", default="benchmarks/skillopt/skillopt/envs/alfworld/skills/initial.md")
    parser.add_argument("--out-dir", default="research/steering/runs/alfworld_hf_rollout_latest")
    parser.add_argument("--mode", choices=["baseline", "steered", "both"], default="both")
    parser.add_argument("--max-episodes", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=-40.0)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--steer-token-slice", choices=["all", "last"], default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-thinking-template", choices=["auto", "true", "false"], default="false")
    args = parser.parse_args()

    os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
    os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args.results_jsonl, max_episodes=args.max_episodes, offset=args.offset)
    skill_content = load_skill(args.skill)
    model, tokenizer = load_model(args.model_path, args.device)
    enable_thinking = None
    if args.enable_thinking_template == "true":
        enable_thinking = True
    elif args.enable_thinking_template == "false":
        enable_thinking = False

    summary: dict[str, Any] = {
        "created_at_unix": time.time(),
        "model_path": args.model_path,
        "vector_path": args.vector_path,
        "alpha": args.alpha,
        "layer": args.layer,
        "results_jsonl": args.results_jsonl,
        "max_episodes": args.max_episodes,
        "max_steps": args.max_steps,
        "steer_token_slice": args.steer_token_slice,
        "modes": {},
    }

    if args.mode in ("baseline", "both"):
        baseline = run_suite(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            mode="baseline",
            steering=None,
            skill_content=skill_content,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            seed=args.seed,
            enable_thinking=enable_thinking,
            out_dir=out_dir,
        )
        summary["modes"]["baseline"] = summarize_suite(baseline)

    if args.mode in ("steered", "both"):
        token_slice = slice(-1, None) if args.steer_token_slice == "last" else None
        steering = load_steering_vector(args.vector_path, alpha=args.alpha, layer=args.layer, token_slice=token_slice)
        summary["layer"] = steering.layer
        steered = run_suite(
            model=model,
            tokenizer=tokenizer,
            cases=cases,
            mode="steered",
            steering=steering,
            skill_content=skill_content,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=args.device,
            seed=args.seed,
            enable_thinking=enable_thinking,
            out_dir=out_dir,
        )
        summary["modes"]["steered"] = summarize_suite(steered)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
