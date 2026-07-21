#!/usr/bin/env python3
"""Extract paired prompt-contrast vectors from a model-independent dataset.

The script owns experiment concerns (files, sharding and conditioning text).
The extraction rule itself lives in ``research.steering.core.extraction``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from research.steering.adapters.conditioning.reflection import reflection_prefix
from research.steering.adapters.conditioning.skillopt import apply_skillopt_edit, skill_prompt
from research.steering.adapters.models.hf_causal import HFHiddenStateProvider
from research.steering.adapters.models.loading import load_causal_lm
from research.steering.core.extraction.prompt_contrast import PromptContrastExtractor

ALFWORLD_SYSTEM = "You are an expert agent operating in the ALFRED Embodied Environment."


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def conditioned_prompt(unit: dict, state_text: str) -> str:
    application = unit.get("application", "reflection_prefix")
    if application == "reflection_prefix":
        return reflection_prefix(unit["text"]) + state_text
    if application == "skill_prompt":
        return skill_prompt(unit["text"]) + "\n" + state_text
    if application == "skillopt_edit":
        edited = apply_skillopt_edit(unit["base_skill"], unit["edit"])
        return skill_prompt(edited) + "\n" + state_text
    raise ValueError(f"unsupported conditioning application: {application}")


def base_prompt(unit: dict, state_text: str) -> str:
    application = unit.get("application", "reflection_prefix")
    if application == "skillopt_edit":
        return skill_prompt(unit["base_skill"]) + "\n" + state_text
    return state_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--state-field", default="obs_text")
    parser.add_argument("--state-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", choices=("last_token", "mean"), default="last_token")
    parser.add_argument("--shard", default="0/1", help="zero-based INDEX/COUNT")
    parser.add_argument("--system-prompt", default=ALFWORLD_SYSTEM)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--split", choices=("train", "dev", "test", "all"), default="all")
    parser.add_argument(
        "--subset", choices=("all", "text_success", "paired_effective"), default="text_success"
    )
    args = parser.parse_args()

    shard_index, shard_count = map(int, args.shard.split("/"))
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard must satisfy 0 <= INDEX < COUNT")

    units = read_jsonl(args.units)
    if args.split != "all":
        units = [unit for unit in units if unit.get("split") == args.split]
    if args.subset != "all":
        units = [unit for unit in units if bool(unit.get(args.subset))]
    units = [unit for i, unit in enumerate(units) if i % shard_count == shard_index]
    states = read_jsonl(args.states)
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(states), generator=generator)[: args.state_count].tolist()
    state_texts = [states[i][args.state_field] for i in order]

    model, tokenizer = load_causal_lm(args.model_path, args.device)
    provider = HFHiddenStateProvider(
        model,
        tokenizer,
        device=args.device,
        system_prompt=args.system_prompt,
        enable_thinking=args.enable_thinking,
    )
    text_provider = HFHiddenStateProvider(
        model, tokenizer, device=args.device, chat_template=False
    )
    extractor = PromptContrastExtractor(pooling=args.pooling, keep_state_deltas=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = torch.load(args.output, weights_only=False) if args.output.exists() else {}
    base_cache: dict[str, torch.Tensor] = {}
    for unit in units:
        unit_id = unit["unit_id"]
        if unit_id in records:
            continue
        bases = [base_prompt(unit, text) for text in state_texts]
        conditioned = [conditioned_prompt(unit, text) for text in state_texts]
        cache_key = unit.get("base_skill", "__plain_state__")
        if cache_key not in base_cache:
            base_cache[cache_key] = provider.encode(bases, pooling=args.pooling).float()
        base_reps = base_cache[cache_key]
        conditioned_reps = provider.encode(conditioned, pooling=args.pooling).float()
        result = extractor.extract_representations(base_reps, conditioned_reps)
        delta_norm = result.state_deltas.norm(dim=-1)
        base_norm = base_reps.norm(dim=-1).clamp_min(1e-12)
        natural_rho = delta_norm / base_norm
        text_mean = text_provider.encode([unit["text"]], pooling="mean")[0]
        text_last = text_provider.encode([unit["text"]], pooling="last_token")[0]
        records[unit_id] = {
            **{k: v for k, v in unit.items() if k not in {"base_skill", "edit"}},
            "model_id": args.model_id,
            "method": extractor.name,
            "pooling": args.pooling,
            "state_indices": order,
            "layers": result.layers,
            "vector": result.mean_vector.half(),
            "consistency": result.consistency.half(),
            "natural_rho_median": natural_rho.median(dim=0).values.half(),
            "natural_rho_q10": torch.quantile(natural_rho, 0.1, dim=0).half(),
            "natural_rho_q90": torch.quantile(natural_rho, 0.9, dim=0).half(),
            "delta_norm_q10": torch.quantile(delta_norm, 0.1, dim=0).half(),
            "delta_norm_q90": torch.quantile(delta_norm, 0.9, dim=0).half(),
            "base_norm_median": base_norm.median(dim=0).values.half(),
            "text_mean": text_mean.half(),
            "text_last": text_last.half(),
        }
        torch.save(records, args.output)
        print(f"{len(records)}/{len(units)} {unit_id}", flush=True)


if __name__ == "__main__":
    main()
