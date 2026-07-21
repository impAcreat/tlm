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
