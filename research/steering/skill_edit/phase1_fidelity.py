from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from research.steering.rollouts.rollout_hf import (
    SteeringSpec,
    build_messages,
    generate_response,
    normalize_model_response,
)
from research.steering.rollouts.run_alfworld_hf_rollout import (
    SKILLOPT_ROOT,
    load_model,
    load_skill,
    read_jsonl,
    run_suite,
    summarize_suite,
)
from research.steering.skill_edit.intra_step_mean import mean_pairwise_cosine, normalize


DEFAULT_TASK_TYPES = (
    "look_at_obj_in_light",
    "pick_and_place",
    "pick_two_obj_and_place",
)


def select_cases_by_type(
    results_jsonl: str | Path,
    task_types: tuple[str, ...],
    max_per_type: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for task_type in task_types:
        matches = [
            row for row in read_jsonl(results_jsonl)
            if row.get("gamefile") and str(row.get("task_type")) == task_type
        ]
        if not matches:
            raise ValueError(f"no cases found for task_type={task_type}")
        selected.extend(matches[:max_per_type])
    return selected


def _safe_id(value: str) -> str:
    return value.replace(":", "_").replace("/", "_")


def read_trace(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def locate_first_forks(
    out_dir: Path,
    minus_results: list[dict[str, Any]],
    plus_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    plus_index = {str(row["gamefile"]): row for row in plus_results}
    forks: list[dict[str, Any]] = []
    for minus in minus_results:
        plus = plus_index.get(str(minus["gamefile"]))
        if plus is None:
            continue
        task_id = _safe_id(str(minus["id"]))
        minus_trace = read_trace(out_dir / "minus" / task_id / "conversation.json")
        plus_trace = read_trace(out_dir / "plus" / task_id / "conversation.json")
        for step, (left, right) in enumerate(zip(minus_trace, plus_trace)):
            if str(left.get("action") or "").strip() == str(right.get("action") or "").strip():
                continue
            minus_prompt = str(left.get("prompt") or "")
            plus_prompt = str(right.get("prompt") or "")
            forks.append(
                {
                    "id": minus["id"],
                    "gamefile": minus["gamefile"],
                    "task_type": minus["task_type"],
                    "task_description": minus.get("task_description", ""),
                    "step": step,
                    "state_matched": bool(minus_prompt) and minus_prompt == plus_prompt,
                    "prompt": minus_prompt,
                    "minus_action": str(left.get("action") or "").strip(),
                    "plus_action": str(right.get("action") or "").strip(),
                    "minus_response": str(left.get("response") or ""),
                    "plus_response": str(right.get("response") or ""),
                    "minus_hard": int(minus.get("hard") or 0),
                    "plus_hard": int(plus.get("hard") or 0),
                }
            )
            break
    return forks


def _chat_ids(tokenizer, messages: list[dict[str, str]], enable_thinking: bool) -> torch.Tensor:
    kwargs = {
        "conversation": messages,
        "add_generation_prompt": False,
        "return_tensors": "pt",
    }
    kwargs["enable_thinking"] = enable_thinking
    try:
        value = tokenizer.apply_chat_template(**kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        value = tokenizer.apply_chat_template(**kwargs)
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "input_ids"):
        return value.input_ids
    raise TypeError("chat template did not return token ids")


def _last_subsequence(sequence: list[int], target: list[int]) -> tuple[int, int]:
    if not target:
        raise ValueError("empty action token sequence")
    for start in range(len(sequence) - len(target), -1, -1):
        if sequence[start:start + len(target)] == target:
            return start, start + len(target)
    raise ValueError("action tokens not found in teacher-forced chat sequence")


def encode_action_all_layers(
    model,
    tokenizer,
    *,
    prompt: str,
    skill_content: str,
    action: str,
    device: str,
    enable_thinking: bool,
) -> torch.Tensor:
    assistant = f"<action>{action}</action>"
    messages = build_messages(prompt, skill_content=skill_content)
    messages.append({"role": "assistant", "content": assistant})
    input_ids = _chat_ids(tokenizer, messages, enable_thinking).to(device)
    action_ids = tokenizer(action, add_special_tokens=False).input_ids
    start, end = _last_subsequence(input_ids[0].tolist(), list(action_ids))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
    return torch.stack([
        hidden[0, start:end].detach().float().mean(dim=0).cpu()
        for hidden in outputs.hidden_states[1:]
    ])


def extract_vectors(
    model,
    tokenizer,
    forks: list[dict[str, Any]],
    *,
    skill_content: str,
    device: str,
    enable_thinking: bool,
) -> tuple[dict[int, torch.Tensor], dict[str, Any], torch.Tensor]:
    node_vectors: list[torch.Tensor] = []
    used: list[dict[str, Any]] = []
    for fork in forks:
        if not fork["state_matched"]:
            continue
        h_pp = encode_action_all_layers(
            model, tokenizer, prompt=fork["prompt"], skill_content=skill_content,
            action=fork["plus_action"], device=device, enable_thinking=enable_thinking,
        )
        h_mp = encode_action_all_layers(
            model, tokenizer, prompt=fork["prompt"], skill_content="",
            action=fork["plus_action"], device=device, enable_thinking=enable_thinking,
        )
        h_pm = encode_action_all_layers(
            model, tokenizer, prompt=fork["prompt"], skill_content=skill_content,
            action=fork["minus_action"], device=device, enable_thinking=enable_thinking,
        )
        h_mm = encode_action_all_layers(
            model, tokenizer, prompt=fork["prompt"], skill_content="",
            action=fork["minus_action"], device=device, enable_thinking=enable_thinking,
        )
        node_vectors.append(0.5 * ((h_pp - h_mp) + (h_pm - h_mm)))
        used.append({key: fork[key] for key in (
            "id", "gamefile", "task_type", "step", "plus_action", "minus_action",
            "minus_hard", "plus_hard",
        )})
    if not node_vectors:
        raise ValueError("no state-matched fork nodes available for extraction")
    stacked = torch.stack(node_vectors)  # [nodes, layers, hidden]
    vectors = {layer: normalize(stacked[:, layer].mean(dim=0)) for layer in range(stacked.shape[1])}
    consistency = {
        str(layer): (
            float(mean_pairwise_cosine(stacked[:, layer])) if len(stacked) > 1 else None
        )
        for layer in range(stacked.shape[1])
    }
    summary = {
        "nodes": len(stacked),
        "layers": stacked.shape[1],
        "node_metadata": used,
        "node_vector_mean_pairwise_cosine_by_layer": consistency,
    }
    return vectors, summary, stacked


def choose_layers(
    summary: dict[str, Any],
    *,
    layer_min: int,
    layer_max: int,
    top_k: int,
) -> list[int]:
    scores = summary["node_vector_mean_pairwise_cosine_by_layer"]
    valid = [
        (int(layer), float(score))
        for layer, score in scores.items()
        if score is not None and layer_min <= int(layer) <= layer_max
    ]
    if not valid:
        return [max(0, int(summary["layers"]) // 2)]
    return [layer for layer, _ in sorted(valid, key=lambda item: item[1], reverse=True)[:top_k]]


def stratified_fork_split(forks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extraction: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    task_types = list(dict.fromkeys(str(fork["task_type"]) for fork in forks))
    for task_type in task_types:
        group = [fork for fork in forks if str(fork["task_type"]) == task_type]
        if len(group) == 1:
            extraction.extend(group)
            continue
        extraction.extend(group[::2])
        validation.extend(group[1::2])
    return extraction, validation or extraction


def validate_at_forks(
    model,
    tokenizer,
    forks: list[dict[str, Any]],
    *,
    vector: torch.Tensor,
    layer: int,
    alphas: list[float],
    device: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for fork in forks:
        if not fork["state_matched"]:
            continue
        messages = build_messages(fork["prompt"], skill_content="")
        for alpha in alphas:
            response = generate_response(
                model,
                tokenizer,
                messages,
                device=device,
                steering=SteeringSpec(
                    layer=layer,
                    vector=vector,
                    alpha=alpha,
                    token_slice=slice(-1, None),
                ),
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
            )
            from research.steering.rollouts.rollout_hf import extract_action
            action = str(extract_action(normalize_model_response(response)) or "").strip()
            rows.append(
                {
                    "id": fork["id"],
                    "task_type": fork["task_type"],
                    "alpha": alpha,
                    "action": action,
                    "target_action": fork["plus_action"],
                    "baseline_action": fork["minus_action"],
                    "target_recovered": action == fork["plus_action"],
                    "baseline_preserved": action == fork["minus_action"],
                }
            )
    by_alpha = {}
    for alpha in alphas:
        subset = [row for row in rows if row["alpha"] == alpha]
        by_alpha[str(alpha)] = {
            "nodes": len(subset),
            "target_recovered": sum(row["target_recovered"] for row in subset),
            "baseline_preserved": sum(row["baseline_preserved"] for row in subset),
            "changed_other": sum(
                not row["target_recovered"] and not row["baseline_preserved"] for row in subset
            ),
        }
    return {"layer": layer, "by_alpha": by_alpha, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-1 no-skill to best-skill fidelity smoke.")
    parser.add_argument("--results-jsonl", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--model-path", default="models/Qwen3.5-4B")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--task-types", default=",".join(DEFAULT_TASK_TYPES))
    parser.add_argument("--max-per-type", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--alphas", default="0,1,3,5")
    parser.add_argument("--layer-min", type=int, default=8)
    parser.add_argument("--layer-max", type=int, default=27)
    parser.add_argument("--top-layers", type=int, default=4)
    parser.add_argument("--reuse-rollouts", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("ALFWORLD_DATA", str(SKILLOPT_ROOT / "data" / "alfworld_data"))
    os.environ.setdefault("ALFWORLD_WORKER_START_METHOD", "spawn")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task_types = tuple(part.strip() for part in args.task_types.split(",") if part.strip())
    cases = select_cases_by_type(args.results_jsonl, task_types, args.max_per_type)
    skill_content = load_skill(args.skill)
    model, tokenizer = load_model(args.model_path, args.device)

    if args.reuse_rollouts:
        minus = read_jsonl(out_dir / "minus" / "results.jsonl")
        plus = read_jsonl(out_dir / "plus" / "results.jsonl")
    else:
        minus = run_suite(
            model=model, tokenizer=tokenizer, cases=cases, mode="minus", steering=None,
            skill_content="", max_steps=args.max_steps, max_new_tokens=args.max_new_tokens,
            temperature=0.0, device=args.device, seed=args.seed, enable_thinking=False,
            out_dir=out_dir,
        )
        plus = run_suite(
            model=model, tokenizer=tokenizer, cases=cases, mode="plus", steering=None,
            skill_content=skill_content, max_steps=args.max_steps, max_new_tokens=args.max_new_tokens,
            temperature=0.0, device=args.device, seed=args.seed, enable_thinking=False,
            out_dir=out_dir,
        )
    forks = locate_first_forks(out_dir, minus, plus)
    matched = [fork for fork in forks if fork["state_matched"]]
    extraction_forks, validation_forks = stratified_fork_split(matched)
    vectors, extraction, node_vectors = extract_vectors(
        model, tokenizer, extraction_forks, skill_content=skill_content,
        device=args.device, enable_thinking=False,
    )
    candidate_layers = choose_layers(
        extraction,
        layer_min=args.layer_min,
        layer_max=args.layer_max,
        top_k=args.top_layers,
    )
    artifact = {
        "vectors": vectors,
        "node_vectors": node_vectors,
        "candidate_layers": candidate_layers,
        "method": "first_fork_2x2_action_span_context_main_effect",
    }
    torch.save(artifact, out_dir / "skill_vectors.pt")
    validation = {
        str(layer): validate_at_forks(
            model, tokenizer, validation_forks, vector=vectors[layer], layer=layer,
            alphas=[float(part) for part in args.alphas.split(",") if part.strip()],
            device=args.device, max_new_tokens=args.max_new_tokens, enable_thinking=False,
        )
        for layer in candidate_layers
    }
    summary = {
        "created_at_unix": time.time(),
        "skill": args.skill,
        "task_types": task_types,
        "cases": len(cases),
        "minus": summarize_suite(minus),
        "plus": summarize_suite(plus),
        "first_forks": len(forks),
        "state_matched_forks": len(matched),
        "extraction": extraction,
        "validation": validation,
    }
    (out_dir / "forks.json").write_text(json.dumps(forks, indent=2, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
