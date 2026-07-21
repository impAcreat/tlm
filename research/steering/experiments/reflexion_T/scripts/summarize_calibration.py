#!/usr/bin/env python3
"""Summarize matched Dev causal calibration without selecting on Test data."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_rows(paths: list[Path]) -> list[dict]:
    rows, seen = [], set()
    for path in paths:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            eval_id = row["eval_id"]
            if eval_id in seen:
                raise ValueError(f"duplicate eval_id: {eval_id}")
            seen.add(eval_id)
            rows.append(row)
    return rows


def arm_metrics(rows: list[dict]) -> dict:
    first_actions = [r.get("actions", [""])[0] if r.get("actions") else "" for r in rows]
    counts = Counter(first_actions)
    probabilities = [count / len(first_actions) for count in counts.values()] if first_actions else []
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    return {
        "n": len(rows),
        "success_rate": mean([float(r["hard"]) for r in rows]),
        "runtime_error_rate": mean([float(bool(r.get("runtime_error"))) for r in rows]),
        "format_repair_rate": mean([float(r.get("format_repair_rate", 0.0)) for r in rows]),
        "repeat_rate": mean([float(r.get("repeat_rate", 0.0)) for r in rows]),
        "first_action_max_share": max(probabilities, default=0.0),
        "first_action_entropy": entropy,
        "unique_first_actions": len(counts),
    }


def paired_metrics(candidate: list[dict], reference_by_unit: dict[str, dict]) -> dict:
    pairs = [(row, reference_by_unit[row["unit_id"]]) for row in candidate if row["unit_id"] in reference_by_unit]
    wins = sum(int(row["hard"] and not ref["hard"]) for row, ref in pairs)
    losses = sum(int(ref["hard"] and not row["hard"]) for row, ref in pairs)
    action_changed = []
    for row, ref in pairs:
        row_actions, ref_actions = row.get("actions", []), ref.get("actions", [])
        action_changed.append(bool(row_actions and ref_actions and row_actions[0] != ref_actions[0]))
    return {
        "paired_n": len(pairs),
        "success_delta": mean([float(row["hard"]) - float(ref["hard"]) for row, ref in pairs]),
        "wins": wins,
        "losses": losses,
        "ties": len(pairs) - wins - losses,
        "first_action_change_rate": mean([float(x) for x in action_changed]),
    }


def paired_arm_delta(candidate: list[dict], reference: list[dict]) -> dict:
    reference_by_unit = {row["unit_id"]: row for row in reference}
    pairs = [(row, reference_by_unit[row["unit_id"]]) for row in candidate if row["unit_id"] in reference_by_unit]
    return {
        "paired_n": len(pairs),
        "success_delta": mean([float(row["hard"]) - float(ref["hard"]) for row, ref in pairs]),
        "wins": sum(int(row["hard"] and not ref["hard"]) for row, ref in pairs),
        "losses": sum(int(ref["hard"] and not row["hard"]) for row, ref in pairs),
    }


def is_safe(metrics: dict, baseline: dict, *, error_margin: float, rate_margin: float,
            collapse_margin: float) -> bool:
    return (
        metrics["runtime_error_rate"] <= baseline["runtime_error_rate"] + error_margin
        and metrics["format_repair_rate"] <= baseline["format_repair_rate"] + rate_margin
        and metrics["repeat_rate"] <= baseline["repeat_rate"] + rate_margin
        and metrics["first_action_max_share"] <= baseline["first_action_max_share"] + collapse_margin
    )


def summarize(rows: list[dict], *, error_margin: float = 0.0, rate_margin: float = 0.10,
              collapse_margin: float = 0.20) -> dict:
    baseline_rows = [r for r in rows if r["arm"] == "baseline"]
    text_rows = [r for r in rows if r["arm"] == "text"]
    baseline_by_unit = {r["unit_id"]: r for r in baseline_rows}
    if len(baseline_by_unit) != len(baseline_rows):
        raise ValueError("baseline must have at most one row per unit")
    baseline = arm_metrics(baseline_rows)
    text = {**arm_metrics(text_rows), **paired_metrics(text_rows, baseline_by_unit)}

    grouped: dict[tuple[int, float, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["arm"] in {"extracted", "random", "mismatched", "predicted"}:
            grouped[(int(row["layer"]), float(row["multiplier"]), row["arm"])].append(row)

    conditions = []
    keys = sorted({(layer, multiplier) for layer, multiplier, _ in grouped})
    for layer, multiplier in keys:
        condition = {"layer": layer, "multiplier": multiplier, "arms": {}}
        arm_rows_by_name = {}
        for arm in ("extracted", "predicted", "random", "mismatched"):
            arm_rows = grouped.get((layer, multiplier, arm), [])
            if not arm_rows:
                continue
            arm_rows_by_name[arm] = arm_rows
            metrics = {**arm_metrics(arm_rows), **paired_metrics(arm_rows, baseline_by_unit)}
            metrics["safe"] = is_safe(
                metrics, baseline, error_margin=error_margin,
                rate_margin=rate_margin, collapse_margin=collapse_margin,
            )
            condition["arms"][arm] = metrics
        extracted = condition["arms"].get("extracted")
        random = condition["arms"].get("random")
        mismatched = condition["arms"].get("mismatched")
        expected_units = set(baseline_by_unit)
        condition["coverage_complete"] = bool(
            extracted and random and mismatched
            and {r["unit_id"] for r in arm_rows_by_name["extracted"]} == expected_units
            and {r["unit_id"] for r in arm_rows_by_name["random"]} == expected_units
            and {r["unit_id"] for r in arm_rows_by_name["mismatched"]} == expected_units
        )
        condition["extracted_vs_random"] = (
            paired_arm_delta(arm_rows_by_name["extracted"], arm_rows_by_name["random"])
            if extracted and random else None
        )
        condition["extracted_vs_mismatched"] = (
            paired_arm_delta(arm_rows_by_name["extracted"], arm_rows_by_name["mismatched"])
            if extracted and mismatched else None
        )
        condition["content_specific"] = bool(
            condition["coverage_complete"]
            and condition["extracted_vs_random"]["success_delta"] > 0
            and condition["extracted_vs_mismatched"]["success_delta"] > 0
            and extracted["wins"] >= extracted["losses"]
        )
        conditions.append(condition)

    conditions.sort(key=lambda x: (
        not bool(x.get("content_specific")),
        -x.get("arms", {}).get("extracted", {}).get("success_delta", -1.0),
        x["layer"], x["multiplier"],
    ))
    return {
        "scope": "dev_only",
        "primary_metric": "matched_task_success_delta",
        "baseline": baseline,
        "text_upper_bound": text,
        "safety_margins": {
            "runtime_error_rate": error_margin,
            "format_and_repeat_rate": rate_margin,
            "first_action_max_share": collapse_margin,
        },
        "conditions": conditions,
    }


def markdown(summary: dict) -> str:
    baseline, text = summary["baseline"], summary["text_upper_bound"]
    lines = [
        "# Dev causal calibration",
        "",
        "Primary criterion: matched task-success change. Geometry is not used for final selection.",
        "",
        f"- Baseline: {baseline['success_rate']:.3f} ({baseline['n']} units)",
        f"- Text upper bound: {text['success_rate']:.3f}; paired delta {text['success_delta']:+.3f}",
        "",
        "| layer | multiplier | arm | n | success | paired delta | W/L | safe | content-specific |",
        "|---:|---:|:---|---:|---:|---:|:---:|:---:|:---:|",
    ]
    for condition in summary["conditions"]:
        for arm, metrics in condition["arms"].items():
            lines.append(
                f"| {condition['layer']} | {condition['multiplier']:g} | {arm} | {metrics['n']} | "
                f"{metrics['success_rate']:.3f} | {metrics['success_delta']:+.3f} | "
                f"{metrics['wins']}/{metrics['losses']} | {str(metrics['safe']).lower()} | "
                f"{str(condition['content_specific'] if arm == 'extracted' else False).lower()} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--error-margin", type=float, default=0.0)
    parser.add_argument("--rate-margin", type=float, default=0.10)
    parser.add_argument("--collapse-margin", type=float, default=0.20)
    args = parser.parse_args()
    result = summarize(
        load_rows(args.inputs), error_margin=args.error_margin,
        rate_margin=args.rate_margin, collapse_margin=args.collapse_margin,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    output_md = args.output_md or args.output_json.with_suffix(".md")
    output_md.write_text(markdown(result))
    print(json.dumps({"json": str(args.output_json), "markdown": str(output_md),
                      "conditions": len(result["conditions"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
