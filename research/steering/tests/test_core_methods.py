from __future__ import annotations

import numpy as np
import torch

from research.steering.core.application import AdditiveSteerer, generation_only
from research.steering.core.compiler import RidgeCompiler
from research.steering.core.extraction import PromptContrastExtractor
from research.steering.core.metrics.causal import paired_flips
from research.steering.analysis.dose_response import rank_doses
from research.steering.analysis.layer_selection import (
    diversified_pareto_shortlist,
    pareto_front,
    rank_causal_layers,
)
from research.steering.experiments.reflexion_T.scripts.train_compiler import (
    nested_grouped_predictions,
)


class Provider:
    def encode(self, prompts, *, pooling):
        offset = 2.0 if prompts[0].startswith("conditioned") else 0.0
        return torch.arange(len(prompts) * 3 * 4).reshape(len(prompts), 3, 4).float() + offset


def test_prompt_contrast():
    result = PromptContrastExtractor(keep_state_deltas=True).extract(
        Provider(), ["base-a", "base-b"], ["conditioned-a", "conditioned-b"]
    )
    assert result.mean_vector.shape == (3, 4)
    assert torch.allclose(result.mean_vector, torch.full((3, 4), 2.0))
    assert torch.allclose(result.consistency, torch.ones(3))


def test_prompt_contrast_precomputed_layer_subset():
    base = torch.zeros(2, 3, 4)
    conditioned = base.clone()
    conditioned[:, 1] = 3.0
    result = PromptContrastExtractor().extract_representations(
        base, conditioned, layers=[1]
    )
    assert result.layers == (1,)
    assert result.mean_vector.shape == (1, 4)
    assert torch.allclose(result.mean_vector, torch.full((1, 4), 3.0))


class Block(torch.nn.Module):
    def forward(self, x):
        return x


class TupleBlock(torch.nn.Module):
    def forward(self, x):
        return x, "cache"


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([Block()])


def test_generation_only_additive():
    model = Toy()
    resolver = lambda model, layer: model.layers[layer]
    with AdditiveSteerer(model, resolver, 0, torch.ones(4), 2.0, generation_only):
        decode = model.layers[0](torch.zeros(1, 1, 4))
        prefill = model.layers[0](torch.zeros(1, 3, 4))
    assert torch.allclose(decode, torch.full_like(decode, 2.0))
    assert torch.allclose(prefill, torch.zeros_like(prefill))


def test_additive_preserves_tuple_outputs():
    model = Toy()
    model.layers[0] = TupleBlock()
    resolver = lambda model, layer: model.layers[layer]
    with AdditiveSteerer(model, resolver, 0, torch.ones(4), 2.0):
        hidden, cache = model.layers[0](torch.zeros(1, 2, 4))
    assert cache == "cache"
    assert torch.allclose(hidden, torch.full_like(hidden, 2.0))


def test_ridge_compiler_and_flips():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 5)).astype(np.float32)
    weight = rng.normal(size=(5, 3)).astype(np.float32)
    y = x @ weight
    pred = RidgeCompiler(alpha=1e-4).fit(x, y).predict(x)
    assert np.mean((pred - y) ** 2) < 1e-6
    assert paired_flips({"a": 0, "b": 1}, {"a": 1, "b": 1})["treatment_only"] == 1


def test_causal_layer_and_dose_ranking_prioritize_safe_task_gain():
    rows = []
    for layer, extracted, random in ((10, (1, 1), (0, 0)), (20, (1, 1), (1, 1))):
        for task, baseline in zip(("a", "b"), (0, 0)):
            rows.extend(
                [
                    {"layer": layer, "arm": "baseline", "task_id": task, "success": baseline},
                    {"layer": layer, "arm": "extracted", "task_id": task, "success": extracted[0]},
                    {"layer": layer, "arm": "random", "task_id": task, "success": random[0]},
                ]
            )
    assert rank_causal_layers(rows)[0]["layer"] == 10
    doses = [
        {"dose": 1.0, "success": 1, "invalid_rate": 0.0},
        {"dose": 2.0, "success": 1, "invalid_rate": 0.4},
    ]
    assert rank_doses(doses, max_invalid_rate=0.1)[0]["dose"] == 1.0


def test_pareto_shortlist_balances_metrics_and_depth():
    rows = [
        {"layer": 0, "cross_state_consistency": 0.9, "heldout_T_residual_cos": 0.9,
         "unit_specific_ratio": 0.1, "shared_component_ratio": 0.9},
        {"layer": 1, "cross_state_consistency": 0.7, "heldout_T_residual_cos": 0.7,
         "unit_specific_ratio": 0.7, "shared_component_ratio": 0.3},
        {"layer": 2, "cross_state_consistency": 0.6, "heldout_T_residual_cos": 0.6,
         "unit_specific_ratio": 0.6, "shared_component_ratio": 0.4},
        {"layer": 3, "cross_state_consistency": 0.8, "heldout_T_residual_cos": 0.8,
         "unit_specific_ratio": 0.8, "shared_component_ratio": 0.2},
    ]
    front = pareto_front(
        rows,
        maximize=("cross_state_consistency", "heldout_T_residual_cos", "unit_specific_ratio"),
        minimize=("shared_component_ratio",),
    )
    assert {row["layer"] for row in front} == {0, 3}
    result = diversified_pareto_shortlist(rows, depth_bins=2)
    assert [row["layer"] for row in result["candidates"]] == [1, 3]


def test_nested_compiler_predictions_cover_outer_rows_without_group_leakage():
    rng = np.random.default_rng(3)
    groups = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"]), 2)
    x = rng.normal(size=(len(groups), 4)).astype(np.float32)
    y = np.stack((x[:, 0] + x[:, 1], x[:, 2] - x[:, 3]), axis=1).astype(np.float32)
    result = nested_grouped_predictions(
        x, y, groups, alphas=[1e-4, 1.0], outer_folds=3, inner_folds=2,
    )
    assert result["predictions"].shape == y.shape
    assert result["shuffled_predictions"].shape == y.shape
    assert set(result["fold_ids"]) == {0, 1, 2}
    assert len(result["selected_alphas"]) == 3
