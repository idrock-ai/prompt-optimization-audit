import json
from analysis.dose_response import collect

def _mk(tmp, mt, acc_ona, trunc):
    d = tmp / f"mt{mt}"; d.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(10):
        items.append({"model": "m", "condition": "cot", "subject": "ona_tili", "qid": i,
                      "correct": "A", "is_correct": 1, "parse_error": False,
                      "truncated": False, "predicted": "A", "rescue_correct": 1,
                      "max_tokens": mt})
        items.append({"model": "m", "condition": "dspy_bootstrap", "subject": "ona_tili",
                      "qid": i, "correct": "A", "is_correct": int(i < acc_ona),
                      "parse_error": False, "truncated": i < trunc, "predicted": "A",
                      "rescue_correct": int(i < acc_ona), "max_tokens": mt})
    (d / "m_items.jsonl").write_text("\n".join(json.dumps(x) for x in items))

def test_collect_trend(tmp_path):
    _mk(tmp_path, 256, 4, 6); _mk(tmp_path, 512, 6, 4)
    _mk(tmp_path, 1024, 8, 2); _mk(tmp_path, 2048, 10, 0)
    table = collect(str(tmp_path), budgets=(256, 512, 1024, 2048))
    row = table["m"]
    assert [row[b]["trunc_boot"] for b in (256, 512, 1024, 2048)] == [6, 4, 2, 0]
    assert row["trend_p"] < 0.05
    assert row["monotone"] is True


def test_monotonicity_is_reported_not_assumed(tmp_path):
    """The truncation series can trend significantly without falling at every step --
    qwen3.5:9b's real series is 13/6/3/5. The flag exists so the paper cannot describe
    a non-monotone series as monotone."""
    _mk(tmp_path, 256, 4, 8); _mk(tmp_path, 512, 6, 4)
    _mk(tmp_path, 1024, 8, 1); _mk(tmp_path, 2048, 10, 3)
    row = collect(str(tmp_path), budgets=(256, 512, 1024, 2048))["m"]
    assert row["monotone"] is False


def test_outcome_test_is_paired_on_the_accuracy_not_the_truncations(tmp_path):
    """The trend test only shows the knob turned. The outcome test asks whether native
    accuracy improved, paired item-by-item between the budget extremes."""
    _mk(tmp_path, 256, 4, 6); _mk(tmp_path, 2048, 10, 0)
    row = collect(str(tmp_path), budgets=(256, 2048))["m"]
    o = row["outcome"]
    assert o["lo_budget"] == 256 and o["hi_budget"] == 2048 and o["n_paired"] == 10
    # items 4..9 are wrong at 256 and right at 2048; none regress
    assert o["c_lo_wrong_hi_right"] == 6 and o["b_lo_right_hi_wrong"] == 0
    assert o["p"] < 0.05
    assert row["residual"] == {"budget": 2048, "delta": 0.0, "truncations": 0}


def test_outcome_test_absent_for_a_single_budget_cell(tmp_path):
    """A model present at one budget must not be silently compared against itself."""
    _mk(tmp_path, 512, 6, 4)
    assert collect(str(tmp_path), budgets=(512,))["m"]["outcome"] is None
