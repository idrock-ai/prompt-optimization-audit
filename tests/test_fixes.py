from analysis.fixes_table import recovery, retention, summarise


def test_recovery_arithmetic():
    # cot ona=36, vanilla boot ona=27, fix ona=34 -> recovery 7/9
    assert abs(recovery(cot=36.0, vanilla=27.0, fixed=34.0) - 7 / 9 * 100) < 1e-6


def test_recovery_is_none_when_nothing_was_lost():
    """The pre-registered estimator scored these as 100, which silently credits the fix
    for models it never had to repair. The function now refuses to decide; the caller
    does, explicitly."""
    assert recovery(cot=36.0, vanilla=36.0, fixed=36.0) is None
    assert recovery(cot=36.0, vanilla=39.0, fixed=30.0) is None   # bootstrap improved


def test_retention_arithmetic_and_none():
    # cot math=70, vanilla=80 (gain 10), fix=78 -> retained 8/10
    assert abs(retention(cot=70.0, vanilla=80.0, fixed=78.0) - 80.0) < 1e-6
    assert retention(cot=70.0, vanilla=70.0, fixed=70.0) is None
    assert retention(cot=70.0, vanilla=65.0, fixed=68.0) is None  # bootstrap lost math


def _row(model, cot_o, van_o, fix_o, cot_m, van_m, fix_m):
    return {"model": model, "cot_ona": cot_o, "vanilla_ona": van_o,
            "cot_math": cot_m, "vanilla_math": van_m,
            "eroded": cot_o - van_o > 0, "math_gained": van_m - cot_m > 0,
            "fixes": {"f": {
                "ona": fix_o, "math": fix_m,
                "recovery": recovery(cot_o, van_o, fix_o),
                "retention": retention(cot_m, van_m, fix_m),
                "ona_points_lost": max(0.0, cot_o - van_o),
                "ona_points_back": (fix_o - van_o) if cot_o - van_o > 0 else 0.0,
                "math_points_gained": max(0.0, van_m - cot_m),
                "math_points_kept": (fix_m - cot_m) if van_m - cot_m > 0 else 0.0}}}


def test_pooled_and_per_model_mean_disagree_as_documented():
    """The two estimators are not interchangeable. One model erodes 10 points and the fix
    returns 5 (a real 50%); the other never eroded, so it contributes a definitional 100
    to the mean and nothing at all to the pooled figure."""
    rows = [_row("eroder", 40.0, 30.0, 35.0, 50.0, 60.0, 58.0),
            _row("untouched", 40.0, 42.0, 42.0, 50.0, 60.0, 60.0)]
    s = summarise(rows)["f"]
    assert s["pooled_recovery"] == 50.0              # 5 points back of 10 lost
    assert s["per_model_mean_recovery"] == 75.0      # (50 + definitional 100) / 2
    assert s["n_definitional_recovery_100"] == 1
    # math: 8 + 10 kept of 10 + 10 gained
    assert s["pooled_retention"] == 90.0


def test_overshoot_inflates_the_mean_but_not_the_pooled_figure():
    """A fix that overshoots CoT on one model scores >100 there and drags the mean up.
    Pooled arithmetic caps nothing but weights by points, so one small model cannot
    carry the summary."""
    rows = [_row("small", 40.0, 39.0, 42.0, 50.0, 60.0, 60.0),   # lost 1, back 3 -> 300%
            _row("big", 40.0, 20.0, 22.0, 50.0, 60.0, 60.0)]     # lost 20, back 2 -> 10%
    s = summarise(rows)["f"]
    assert s["n_recovery_over_100"] == 1
    assert s["per_model_mean_recovery"] == 155.0                 # (300 + 10) / 2
    assert abs(s["pooled_recovery"] - 100 * 5 / 21) < 0.05       # 5 points of 21
    assert s["bar_pooled"] == "miss"
