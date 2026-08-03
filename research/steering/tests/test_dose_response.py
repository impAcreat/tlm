from research.steering.analysis.dose_response import adaptive_dose_value


def row(unit, arm, multiplier, hard):
    return {"unit_id": unit, "arm": arm, "multiplier": multiplier, "hard": hard}


def test_no_dose_heterogeneity_gives_zero_gap():
    """Every unit peaks at the same dose, so a global dose loses nothing."""
    rows = []
    for unit in ("a", "b", "c"):
        rows += [row(unit, "extracted", 0.5, 0), row(unit, "extracted", 2.0, 1)]
    result = adaptive_dose_value(rows, arms=("extracted",))
    arm = result["arms"]["extracted"]
    assert arm["best_global_dose"] == 2.0
    assert arm["best_global_success"] == 1.0
    assert arm["adaptive_gap"] == 0.0


def test_dose_heterogeneity_is_credited_only_above_the_control():
    """Units peak at different doses; the random arm sets the selection-bias floor."""
    rows = [
        row("a", "extracted", 0.5, 1), row("a", "extracted", 2.0, 0),
        row("b", "extracted", 0.5, 0), row("b", "extracted", 2.0, 1),
        # control: same after-the-fact freedom, but the wins are spurious
        row("a", "random", 0.5, 1), row("a", "random", 2.0, 0),
        row("b", "random", 0.5, 0), row("b", "random", 2.0, 0),
    ]
    result = adaptive_dose_value(rows)
    assert result["arms"]["extracted"]["best_global_success"] == 0.5
    assert result["arms"]["extracted"]["oracle_per_unit_success"] == 1.0
    assert result["arms"]["extracted"]["adaptive_gap"] == 0.5
    assert result["arms"]["random"]["adaptive_gap"] == 0.0
    assert result["excess_over_control"]["extracted"] == 0.5


def test_missing_cells_do_not_inflate_the_oracle():
    """A unit absent from a dose must not count as a success at that dose."""
    rows = [
        row("a", "extracted", 0.5, 0), row("a", "extracted", 2.0, 0),
        row("b", "extracted", 0.5, 1),  # b never ran at 2.0
    ]
    result = adaptive_dose_value(rows, arms=("extracted",))
    arm = result["arms"]["extracted"]
    assert arm["n_units"] == 2
    assert arm["per_dose_success"][2.0] == 0.0
    assert arm["oracle_per_unit_success"] == 0.5
