from research.steering.experiments.reflexion_T.scripts.config import deep_merge


def test_deep_merge_keeps_unrelated_fields():
    base = {"model": {"id": "x", "dtype": "bf16"}, "seed": 42}
    merged = deep_merge(base, {"model": {"id": "y"}})
    assert merged == {"model": {"id": "y", "dtype": "bf16"}, "seed": 42}
