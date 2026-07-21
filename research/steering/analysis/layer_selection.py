"""Development-only layer analysis with causal task performance as primary."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

from research.steering.core.compiler import RidgeCompiler
from research.steering.core.metrics.geometry import residual_cosine_rows


def grouped_compiler_diagnostic(text_by_layer, vectors_by_layer, groups, *, alpha=100.0, folds=5):
    """Return residual-cosine diagnostics; do not use these alone to select a layer."""
    text = np.asarray(text_by_layer)
    vectors = np.asarray(vectors_by_layer)
    groups = np.asarray(groups)
    if text.shape != vectors.shape or text.ndim != 3:
        raise ValueError("inputs must be matching [unit, layer, hidden] arrays")
    scores = np.zeros(text.shape[1], dtype=np.float64)
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    splits = list(splitter.split(text, groups=groups))
    for layer in range(text.shape[1]):
        values = []
        for train, test in splits:
            model = RidgeCompiler(alpha).fit(text[train, layer], vectors[train, layer])
            pred = model.predict(text[test, layer])
            values.extend(residual_cosine_rows(pred, vectors[test, layer], vectors[train, layer].mean(0)))
        scores[layer] = np.mean(values)
    return scores


def _dominates(left, right, maximize, minimize):
    left_values = [float(left[key]) for key in maximize] + [-float(left[key]) for key in minimize]
    right_values = [float(right[key]) for key in maximize] + [-float(right[key]) for key in minimize]
    return all(a >= b for a, b in zip(left_values, right_values)) and any(
        a > b for a, b in zip(left_values, right_values)
    )


def pareto_front(rows, *, maximize, minimize=()):
    """Return non-dominated diagnostic rows without collapsing metrics early."""
    rows = list(rows)
    return [
        row for index, row in enumerate(rows)
        if not any(
            _dominates(other, row, maximize, minimize)
            for other_index, other in enumerate(rows)
            if other_index != index
        )
    ]


def diversified_pareto_shortlist(rows, *, depth_bins=8):
    """Choose one balanced Pareto candidate per depth region.

    This is only an offline screen. It deliberately balances stability,
    task-specific signal, compiler predictability, and low shared component;
    Dev causal evaluation must make the final layer choice.
    """
    rows = sorted((dict(row) for row in rows), key=lambda row: int(row["layer"]))
    if not rows:
        return {"pareto_layers": [], "candidates": []}
    maximize = ("cross_state_consistency", "heldout_T_residual_cos", "unit_specific_ratio")
    minimize = ("shared_component_ratio",)
    front = pareto_front(rows, maximize=maximize, minimize=minimize)
    front_layers = {int(row["layer"]) for row in front}

    utility = {}
    for key, sign in [(x, 1.0) for x in maximize] + [(x, -1.0) for x in minimize]:
        values = np.asarray([sign * float(row[key]) for row in rows], dtype=np.float64)
        low, high = float(values.min()), float(values.max())
        normalized = np.full_like(values, 0.5) if high == low else (values - low) / (high - low)
        utility[key] = normalized
    for index, row in enumerate(rows):
        # Maximin prevents one attractive diagnostic (especially cosine) from
        # hiding a weak causal prerequisite.
        row["balanced_min_utility"] = float(min(values[index] for values in utility.values()))

    candidates = []
    for indices in np.array_split(np.arange(len(rows)), min(depth_bins, len(rows))):
        bucket = [rows[int(index)] for index in indices]
        pool = pareto_front(bucket, maximize=maximize, minimize=minimize)
        chosen = max(pool, key=lambda row: (row["balanced_min_utility"], -int(row["layer"])))
        candidates.append({
            "layer": int(chosen["layer"]),
            "depth_start": int(bucket[0]["layer"]),
            "depth_end": int(bucket[-1]["layer"]),
            "pareto_source": (
                "global_pareto" if int(chosen["layer"]) in front_layers else "local_pareto"
            ),
            "balanced_min_utility": chosen["balanced_min_utility"],
        })
    return {
        "pareto_layers": sorted(front_layers),
        "candidates": candidates,
    }


def rank_causal_layers(rows, *, geometry_scores=None, max_invalid_increase=0.05):
    """Rank layers using extracted-vector task gain and random controls.

    Each row needs ``layer``, ``arm``, ``task_id`` and ``success``. Optional
    ``invalid_rate`` is averaged as a safety guard. Geometry is only a final
    tie-breaker after causal task effects.
    """
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        grouped[int(row["layer"])][str(row["arm"])][str(row["task_id"])] = row

    ranked = []
    for layer, arms in grouped.items():
        required = {"baseline", "extracted", "random"}
        if not required.issubset(arms):
            continue
        tasks = set(arms["baseline"]) & set(arms["extracted"]) & set(arms["random"])
        if not tasks:
            continue

        def rate(arm, field="success"):
            return float(np.mean([float(arms[arm][task].get(field, 0.0)) for task in tasks]))

        baseline = rate("baseline")
        extracted = rate("extracted")
        random = rate("random")
        baseline_invalid = rate("baseline", "invalid_rate")
        extracted_invalid = rate("extracted", "invalid_rate")
        random_invalid = rate("random", "invalid_rate")
        safe = extracted_invalid - baseline_invalid <= max_invalid_increase
        ranked.append(
            {
                "layer": layer,
                "n_tasks": len(tasks),
                "baseline_success": baseline,
                "extracted_success": extracted,
                "random_success": random,
                "extracted_gain": extracted - baseline,
                "random_gain": random - baseline,
                "specific_gain": extracted - random,
                "invalid_increase": extracted_invalid - baseline_invalid,
                "random_invalid_increase": random_invalid - baseline_invalid,
                "safe": safe,
                "geometry_diagnostic": (
                    float(geometry_scores[layer]) if geometry_scores is not None else None
                ),
            }
        )
    return sorted(
        ranked,
        key=lambda item: (
            item["safe"],
            item["specific_gain"],
            item["extracted_gain"],
            item["geometry_diagnostic"] if item["geometry_diagnostic"] is not None else -np.inf,
        ),
        reverse=True,
    )


# Compatibility alias for analysis notebooks written before the causal ranking
# was separated from the compiler diagnostic.
grouped_layer_scores = grouped_compiler_diagnostic
