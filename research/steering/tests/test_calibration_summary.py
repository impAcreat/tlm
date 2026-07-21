from research.steering.experiments.reflexion_T.scripts.summarize_calibration import (
    matched_bootstrap,
    summarize,
)


def row(unit, arm, hard, *, layer=-1, multiplier=0.0, action="go", repair=0.0, repeat=0.0):
    return {
        "eval_id": f"{unit}|{arm}|{layer}|{multiplier}",
        "unit_id": unit,
        "arm": arm,
        "layer": layer,
        "multiplier": multiplier,
        "hard": hard,
        "runtime_error": None,
        "format_repair_rate": repair,
        "repeat_rate": repeat,
        "actions": [action],
    }


def test_summary_prioritizes_task_success_and_random_control():
    rows = []
    for unit in ("a", "b"):
        rows += [row(unit, "baseline", 0), row(unit, "text", 1)]
    rows += [
        row("a", "extracted", 1, layer=20, multiplier=0.5, action="open"),
        row("b", "extracted", 1, layer=20, multiplier=0.5, action="take"),
        row("a", "random", 0, layer=20, multiplier=0.5, action="look"),
        row("b", "random", 0, layer=20, multiplier=0.5, action="inventory"),
        row("a", "mismatched", 0, layer=20, multiplier=0.5, action="look"),
        row("b", "mismatched", 0, layer=20, multiplier=0.5, action="inventory"),
    ]
    result = summarize(rows)
    condition = result["conditions"][0]
    assert result["primary_metric"] == "matched_task_success_delta"
    assert condition["arms"]["extracted"]["success_delta"] == 1.0
    assert condition["arms"]["random"]["success_delta"] == 0.0
    assert condition["content_specific"] is True


def test_safety_gate_rejects_repeat_collapse():
    rows = [row("a", "baseline", 0, repeat=0.0), row("a", "text", 1)]
    rows += [
        row("a", "extracted", 1, layer=10, multiplier=1.0, repeat=0.8),
        row("a", "random", 0, layer=10, multiplier=1.0),
        row("a", "mismatched", 0, layer=10, multiplier=1.0),
    ]
    result = summarize(rows)
    assert result["conditions"][0]["arms"]["extracted"]["safe"] is False


def test_matched_bootstrap_reports_task_level_uncertainty():
    certain = matched_bootstrap([1.0, 1.0, 1.0], samples=100)
    assert certain["ci95"] == [1.0, 1.0]
    assert certain["probability_positive"] == 1.0
    empty = matched_bootstrap([])
    assert empty["ci95"] == [None, None]
