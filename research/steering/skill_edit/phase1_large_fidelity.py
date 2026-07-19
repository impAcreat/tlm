from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import torch

from research.steering.rollouts.rollout_hf import (
    SteeringSpec,
    extract_action,
    generate_response,
    normalize_model_response,
)
from research.steering.rollouts.run_alfworld_hf_rollout import load_model, load_skill, read_jsonl
from research.steering.skill_edit.intra_step_mean import mean_pairwise_cosine, normalize


SYSTEM_PROMPT = (
    "You are an expert agent operating in the ALFRED Embodied Environment. "
    "For every turn, state one explicit <intention>...</intention> before "
    "the required <action>...</action>. The intention should name the "
    "immediate environmental result you want this action to verify or achieve."
)


def skill_prompt(skill_content: str) -> str:
    if not skill_content or not skill_content.strip():
        return ""
    return (
        "\n\n## Skill Knowledge\n"
        "Below is a skill document with learned strategies. "
        "Use these guidelines to inform your decisions:\n\n"
        f"{skill_content}\n"
    )


def messages(prompt: str, skill_content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{skill_prompt(skill_content)}{prompt}"},
    ]


def read_trace(root: Path, task_id: str) -> list[dict[str, Any]]:
    path = root / "predictions" / task_id / "conversation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def collect_candidate_forks(
    minus_root: Path,
    plus_roots: dict[str, Path],
) -> list[dict[str, Any]]:
    minus_rows = read_jsonl(minus_root / "results.jsonl")
    plus_indices = {
        task_type: {str(row["id"]): row for row in read_jsonl(root / "results.jsonl")}
        for task_type, root in plus_roots.items()
    }
    forks: list[dict[str, Any]] = []
    for minus in minus_rows:
        task_type = str(minus.get("task_type") or "")
        if task_type not in plus_roots:
            continue
        task_id = str(minus["id"])
        plus = plus_indices[task_type].get(task_id)
        if plus is None:
            continue
        minus_trace = read_trace(minus_root, task_id)
        plus_trace = read_trace(plus_roots[task_type], task_id)
        for step, (left, right) in enumerate(zip(minus_trace, plus_trace)):
            left_action = str(left.get("action") or "").strip()
            right_action = str(right.get("action") or "").strip()
            if left_action == right_action:
                continue
            left_prompt = str(left.get("prompt") or "")
            right_prompt = str(right.get("prompt") or "")
            forks.append(
                {
                    "id": task_id,
                    "task_type": task_type,
                    "gamefile": minus.get("gamefile", ""),
                    "step": step,
                    "prompt": left_prompt,
                    "state_matched": bool(left_prompt) and left_prompt == right_prompt,
                    "vllm_minus_action": left_action,
                    "vllm_plus_action": right_action,
                    "minus_hard": int(minus.get("hard") or 0),
                    "plus_hard": int(plus.get("hard") or 0),
                }
            )
            break
    return forks


def generate_action(
    model,
    tokenizer,
    *,
    prompt: str,
    skill_content: str,
    device: str,
    max_new_tokens: int,
    steering: SteeringSpec | None = None,
) -> tuple[str, str]:
    response = generate_response(
        model,
        tokenizer,
        messages(prompt, skill_content),
        device=device,
        steering=steering,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enable_thinking=False,
    )
    normalized = normalize_model_response(response)
    return str(extract_action(normalized) or "").strip(), response


def probe_local_flips(
    model,
    tokenizer,
    forks: list[dict[str, Any]],
    *,
    skills: dict[str, str],
    device: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    probed: list[dict[str, Any]] = []
    for index, fork in enumerate(forks):
        if not fork["state_matched"]:
            continue
        minus_action, minus_response = generate_action(
            model, tokenizer, prompt=fork["prompt"], skill_content="",
            device=device, max_new_tokens=max_new_tokens,
        )
        plus_action, plus_response = generate_action(
            model, tokenizer, prompt=fork["prompt"], skill_content=skills[fork["task_type"]],
            device=device, max_new_tokens=max_new_tokens,
        )
        row = dict(fork)
        row.update(
            {
                "local_minus_action": minus_action,
                "local_plus_action": plus_action,
                "local_minus_response": minus_response,
                "local_plus_response": plus_response,
                "local_flip": bool(minus_action) and bool(plus_action) and minus_action != plus_action,
            }
        )
        probed.append(row)
        print(json.dumps({"probe": index + 1, "total": len(forks), **row}, ensure_ascii=False), flush=True)
    return probed


def split_nodes(
    nodes: list[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    extraction: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for task_type in sorted({str(node["task_type"]) for node in nodes}):
        group = [node for node in nodes if node["task_type"] == task_type]
        rng.shuffle(group)
        if len(group) < 2:
            extraction.extend(group)
            continue
        cut = min(len(group) - 1, max(1, round(len(group) * train_fraction)))
        extraction.extend(group[:cut])
        validation.extend(group[cut:])
    return extraction, validation or extraction


def _chat_ids(tokenizer, conversation: list[dict[str, str]]) -> torch.Tensor:
    kwargs = {
        "conversation": conversation,
        "add_generation_prompt": False,
        "return_tensors": "pt",
        "enable_thinking": False,
    }
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
    for start in range(len(sequence) - len(target), -1, -1):
        if sequence[start:start + len(target)] == target:
            return start, start + len(target)
    raise ValueError("action token sequence not found")


def encode_action_all_layers(
    model,
    tokenizer,
    *,
    prompt: str,
    skill_content: str,
    action: str,
    device: str,
) -> torch.Tensor:
    conversation = messages(prompt, skill_content)
    conversation.append({"role": "assistant", "content": f"<action>{action}</action>"})
    input_ids = _chat_ids(tokenizer, conversation).to(device)
    action_ids = list(tokenizer(action, add_special_tokens=False).input_ids)
    start, end = _last_subsequence(input_ids[0].tolist(), action_ids)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            output_hidden_states=True,
            use_cache=False,
        )
    return torch.stack([
        hidden[0, start:end].detach().float().mean(dim=0).cpu()
        for hidden in outputs.hidden_states[1:]
    ])


def extract_node_vectors(
    model,
    tokenizer,
    nodes: list[dict[str, Any]],
    *,
    skills: dict[str, str],
    device: str,
) -> torch.Tensor:
    vectors: list[torch.Tensor] = []
    for index, node in enumerate(nodes):
        prompt = node["prompt"]
        skill = skills[node["task_type"]]
        plus_action = node["local_plus_action"]
        minus_action = node["local_minus_action"]
        h_pp = encode_action_all_layers(
            model, tokenizer, prompt=prompt, skill_content=skill, action=plus_action, device=device
        )
        h_mp = encode_action_all_layers(
            model, tokenizer, prompt=prompt, skill_content="", action=plus_action, device=device
        )
        h_pm = encode_action_all_layers(
            model, tokenizer, prompt=prompt, skill_content=skill, action=minus_action, device=device
        )
        h_mm = encode_action_all_layers(
            model, tokenizer, prompt=prompt, skill_content="", action=minus_action, device=device
        )
        vectors.append(0.5 * ((h_pp - h_mp) + (h_pm - h_mm)))
        print(json.dumps({"extract": index + 1, "total": len(nodes), "id": node["id"]}), flush=True)
    return torch.stack(vectors)


def consistency_by_layer(vectors: torch.Tensor) -> list[float | None]:
    return [
        float(mean_pairwise_cosine(vectors[:, layer])) if len(vectors) > 1 else None
        for layer in range(vectors.shape[1])
    ]


def top_layers(scores: list[float | None], layer_min: int, layer_max: int, count: int) -> list[int]:
    ranked = [
        (layer, float(scores[layer]))
        for layer in range(layer_min, min(layer_max + 1, len(scores)))
        if scores[layer] is not None and math.isfinite(float(scores[layer]))
    ]
    return [layer for layer, _ in sorted(ranked, key=lambda item: item[1], reverse=True)[:count]]


def mean_vectors(vectors: torch.Tensor) -> dict[int, torch.Tensor]:
    return {layer: normalize(vectors[:, layer].mean(dim=0)) for layer in range(vectors.shape[1])}


def validate_vector_family(
    model,
    tokenizer,
    nodes: list[dict[str, Any]],
    *,
    vectors: dict[int, torch.Tensor],
    layers: list[int],
    alphas: list[float],
    device: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for layer in layers:
        for alpha in alphas:
            for node in nodes:
                action, _ = generate_action(
                    model,
                    tokenizer,
                    prompt=node["prompt"],
                    skill_content="",
                    device=device,
                    max_new_tokens=max_new_tokens,
                    steering=SteeringSpec(
                        layer=layer,
                        vector=vectors[layer],
                        alpha=alpha,
                        token_slice=slice(-1, None),
                    ),
                )
                rows.append(
                    {
                        "id": node["id"],
                        "task_type": node["task_type"],
                        "layer": layer,
                        "alpha": alpha,
                        "action": action,
                        "target_action": node["local_plus_action"],
                        "baseline_action": node["local_minus_action"],
                        "target_recovered": action == node["local_plus_action"],
                        "baseline_preserved": action == node["local_minus_action"],
                    }
                )
    settings = []
    for layer in layers:
        for alpha in alphas:
            subset = [row for row in rows if row["layer"] == layer and row["alpha"] == alpha]
            settings.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "nodes": len(subset),
                    "target_recovered": sum(row["target_recovered"] for row in subset),
                    "baseline_preserved": sum(row["baseline_preserved"] for row in subset),
                    "changed_other": sum(
                        not row["target_recovered"] and not row["baseline_preserved"] for row in subset
                    ),
                }
            )
    return {"layers": layers, "settings": settings, "rows": rows}


def plot_diagnostics(
    out_dir: Path,
    *,
    global_scores: list[float | None],
    task_scores: dict[str, list[float | None]],
    node_vectors: torch.Tensor,
    node_labels: list[str],
    pca_layer: int,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.load_default()
    palette = [(45, 90, 180), (220, 105, 45), (55, 155, 85), (155, 85, 175)]
    series = [("global", global_scores), *sorted(task_scores.items())]
    image = Image.new("RGB", (1500, 650), "white")
    draw = ImageDraw.Draw(image)
    draw.text((50, 20), "2x2 node-vector consistency by layer", fill="black", font=font)
    box = (70, 55, 1430, 585)
    draw.rectangle(box, outline=(50, 50, 50), width=2)
    finite = [float(value) for _, values in series for value in values if value is not None and math.isfinite(float(value))]
    y_min, y_max = min(finite), max(finite)
    pad = max((y_max - y_min) * 0.08, 0.02)
    y_min, y_max = y_min - pad, y_max + pad
    for grid in range(1, 5):
        y = box[1] + grid * (box[3] - box[1]) / 5
        draw.line((box[0], y, box[2], y), fill=(225, 225, 225), width=1)
    for index, (label, values) in enumerate(series):
        points = []
        for layer, value in enumerate(values):
            if value is None or not math.isfinite(float(value)):
                continue
            x = box[0] + layer / max(1, len(values) - 1) * (box[2] - box[0])
            y = box[3] - (float(value) - y_min) / (y_max - y_min) * (box[3] - box[1])
            points.append((x, y))
        color = palette[index % len(palette)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3 if index == 0 else 2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        draw.text((80 + index * 330, 610), label, fill=color, font=font)
    image.save(out_dir / "layer_consistency.png")

    x = node_vectors[:, pca_layer].float()
    centered = x - x.mean(dim=0, keepdim=True)
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    coords = centered @ vh[:2].T
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((45, 20), f"First-fork 2x2 vectors at layer {pca_layer}", fill="black", font=font)
    box = (70, 55, 830, 620)
    draw.rectangle(box, outline=(50, 50, 50), width=2)
    xs, ys = coords[:, 0].tolist(), coords[:, 1].tolist()
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_pad, y_pad = max((x_max - x_min) * 0.08, 1e-6), max((y_max - y_min) * 0.08, 1e-6)
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = y_min - y_pad, y_max + y_pad
    unique_labels = sorted(set(node_labels))
    color_map = {label: palette[index % len(palette)] for index, label in enumerate(unique_labels)}
    for px, py, label in zip(xs, ys, node_labels):
        sx = box[0] + (px - x_min) / (x_max - x_min) * (box[2] - box[0])
        sy = box[3] - (py - y_min) / (y_max - y_min) * (box[3] - box[1])
        draw.ellipse((sx - 6, sy - 6, sx + 6, sy + 6), fill=color_map[label], outline=(40, 40, 40))
    for index, label in enumerate(unique_labels):
        draw.text((70 + index * 260, 655), label, fill=color_map[label], font=font)
    image.save(out_dir / "node_vector_pca.png")


def parse_mapping(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, path = value.split("=", 1)
        result[key.strip()] = path.strip()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Large no-skill to semantic-skill fidelity study.")
    parser.add_argument("--minus-root", type=Path, required=True)
    parser.add_argument("--plus-root", action="append", required=True)
    parser.add_argument("--skill", action="append", required=True)
    parser.add_argument("--model-path", default="models/Qwen3.5-4B")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--layer-min", type=int, default=8)
    parser.add_argument("--layer-max", type=int, default=27)
    parser.add_argument("--global-top-layers", type=int, default=3)
    parser.add_argument("--task-top-layers", type=int, default=2)
    parser.add_argument("--alphas", default="1,2,3,5")
    args = parser.parse_args()

    plus_roots = {key: Path(path) for key, path in parse_mapping(args.plus_root).items()}
    skill_paths = parse_mapping(args.skill)
    skills = {key: load_skill(path) for key, path in skill_paths.items()}
    if set(plus_roots) != set(skills):
        raise ValueError("plus-root and skill task-type keys must match")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model(args.model_path, args.device)

    candidates = collect_candidate_forks(args.minus_root, plus_roots)
    probed = probe_local_flips(
        model,
        tokenizer,
        candidates,
        skills=skills,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    local_flips = [node for node in probed if node["local_flip"]]
    extraction, validation = split_nodes(
        local_flips,
        seed=args.seed,
        train_fraction=args.train_fraction,
    )
    node_vectors = extract_node_vectors(
        model,
        tokenizer,
        extraction,
        skills=skills,
        device=args.device,
    )
    labels = [str(node["task_type"]) for node in extraction]
    global_scores = consistency_by_layer(node_vectors)
    task_scores = {
        task_type: consistency_by_layer(
            torch.stack([vector for vector, label in zip(node_vectors, labels) if label == task_type])
        )
        for task_type in sorted(set(labels))
    }
    global_vectors = mean_vectors(node_vectors)
    task_vectors = {
        task_type: mean_vectors(
            torch.stack([vector for vector, label in zip(node_vectors, labels) if label == task_type])
        )
        for task_type in sorted(set(labels))
    }
    global_layers = top_layers(
        global_scores, args.layer_min, args.layer_max, args.global_top_layers
    )
    task_layers = {
        task_type: top_layers(scores, args.layer_min, args.layer_max, args.task_top_layers)
        for task_type, scores in task_scores.items()
    }
    alphas = [float(value) for value in args.alphas.split(",") if value.strip()]
    global_validation = validate_vector_family(
        model,
        tokenizer,
        validation,
        vectors=global_vectors,
        layers=global_layers,
        alphas=alphas,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    task_validation = {
        task_type: validate_vector_family(
            model,
            tokenizer,
            [node for node in validation if node["task_type"] == task_type],
            vectors=task_vectors[task_type],
            layers=task_layers[task_type],
            alphas=alphas,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
        for task_type in task_vectors
        if any(node["task_type"] == task_type for node in validation)
    }
    artifact = {
        "node_vectors": node_vectors,
        "node_ids": [node["id"] for node in extraction],
        "node_labels": labels,
        "global_vectors": global_vectors,
        "task_vectors": task_vectors,
        "global_layers": global_layers,
        "task_layers": task_layers,
    }
    torch.save(artifact, args.out_dir / "skill_vectors.pt")
    plot_diagnostics(
        args.out_dir,
        global_scores=global_scores,
        task_scores=task_scores,
        node_vectors=node_vectors,
        node_labels=labels,
        pca_layer=global_layers[0],
    )
    summary = {
        "created_at_unix": time.time(),
        "minus_root": str(args.minus_root),
        "plus_roots": {key: str(path) for key, path in plus_roots.items()},
        "skill_paths": skill_paths,
        "candidate_first_forks": len(candidates),
        "state_matched_forks": sum(node["state_matched"] for node in candidates),
        "local_flip_nodes": len(local_flips),
        "local_flip_by_task": {
            task_type: sum(node["task_type"] == task_type for node in local_flips)
            for task_type in sorted(plus_roots)
        },
        "extraction_nodes": len(extraction),
        "validation_nodes": len(validation),
        "extraction_by_task": {
            task_type: labels.count(task_type) for task_type in sorted(plus_roots)
        },
        "validation_by_task": {
            task_type: sum(node["task_type"] == task_type for node in validation)
            for task_type in sorted(plus_roots)
        },
        "global_consistency_by_layer": global_scores,
        "task_consistency_by_layer": task_scores,
        "global_validation": global_validation,
        "task_validation": task_validation,
    }
    (args.out_dir / "forks_and_local_probes.json").write_text(
        json.dumps(probed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
