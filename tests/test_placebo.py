import json

from analysis.placebo import subject_accuracy
from analysis.interaction import analyse


def _item(model, cond, qid, subject, is_correct):
    return {"model": model, "condition": cond, "subject": subject, "qid": qid,
            "correct": "A", "is_correct": is_correct, "rescue_correct": is_correct,
            "predicted": "A", "parse_error": False, "truncated": False}


def _write(tmp, model, rows):
    (tmp / f"{model}_items.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows))


def test_subject_accuracy_pools_over_models():
    data = {"m1": [_item("m1", "cot", 1, "nat", 1), _item("m1", "cot", 2, "nat", 0)],
            "m2": [_item("m2", "cot", 1, "nat", 1), _item("m2", "cot", 2, "nat", 1)]}
    acc = subject_accuracy(data)
    assert acc["nat"] == (75.0, 4)


def test_rotation_flags_only_the_harmed_subject(tmp_path):
    """Two subjects, one harmed and one helped by the same 'optimizer'. Rotating the
    native label must give a high odds ratio for the harmed subject and its reciprocal
    for the other -- the estimator has no notion of which subject is 'native', which is
    exactly what makes the rotation a placebo."""
    rows = []
    for q in range(10):                       # nat: 8 lost, 2 gained
        rows += [_item("m", "cot", f"n{q}", "nat", 1 if q < 8 else 0),
                 _item("m", "dspy_bootstrap", f"n{q}", "nat", 0 if q < 8 else 1)]
    for q in range(10):                       # oth: 2 lost, 8 gained
        rows += [_item("m", "cot", f"o{q}", "oth", 1 if q < 2 else 0),
                 _item("m", "dspy_bootstrap", f"o{q}", "oth", 0 if q < 2 else 1)]
    _write(tmp_path, "m", rows)

    nat = analyse(str(tmp_path), "nat", n_boot=200)
    oth = analyse(str(tmp_path), "oth", n_boot=200)
    assert nat["pooled"]["native_lost"] == 8 and nat["pooled"]["native_gained"] == 2
    assert nat["mh_odds_ratio"] == 16.0          # (8*8)/(2*2)
    # the two views of the same 2x2 are reciprocals, up to the 3-decimal rounding
    # analyse() applies before returning
    assert abs(nat["mh_odds_ratio"] * oth["mh_odds_ratio"] - 1.0) < 0.01


def test_rotation_returns_one_when_both_subjects_fare_alike(tmp_path):
    rows = []
    for subj in ("nat", "oth"):
        for q in range(20):
            harmful = q < 5
            rows.append(_item("m", "cot", f"{subj}{q}", subj, 1 if q < 10 else 0))
            rows.append(_item("m", "dspy_bootstrap", f"{subj}{q}", subj,
                              0 if harmful else (1 if q < 15 else 0)))
    _write(tmp_path, "m", rows)
    assert abs(analyse(str(tmp_path), "nat", n_boot=200)["mh_odds_ratio"] - 1.0) < 1e-6
