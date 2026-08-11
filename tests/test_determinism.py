from analysis.determinism import compare


def _cot(model, qid, subject, is_correct):
    return {"model": model, "condition": "cot", "subject": subject, "qid": qid,
            "is_correct": is_correct}


def _session(model, results, subject="ona_tili"):
    return {(model, q): _cot(model, q, subject, k) for q, k in enumerate(results)}


def test_churn_and_drift_are_different_quantities():
    """Four items change correctness between sessions but two go each way, so accuracy
    is unmoved. Reporting churn alone overstates the instability; reporting drift alone
    hides it. The paper needs both numbers, so the function returns both."""
    a = _session("m", [1, 1, 0, 0, 1, 1, 0, 0, 1, 1])
    b = _session("m", [0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    r = compare(a, b)["m"]
    assert r["churn"] == 4 and r["churn_pct"] == 40.0
    assert r["net_drift"] == 0.0
    assert r["acc_a"] == r["acc_b"] == 60.0


def test_drift_tracks_asymmetric_flips():
    a = _session("m", [1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    b = _session("m", [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    r = compare(a, b)["m"]
    assert r["churn"] == 3 and r["net_drift"] == -30.0


def test_deterministic_model_has_zero_churn():
    a = _session("m", [1, 0, 1, 0])
    r = compare(a, dict(a))["m"]
    assert r["churn"] == 0 and r["net_drift"] == 0.0


def test_subject_filter_and_shared_items_only():
    a = {**_session("m", [1, 1], "ona_tili"),
         ("m", "x0"): _cot("m", "x0", "matematika", 1)}
    b = {**_session("m", [0, 1], "ona_tili"),
         ("m", "x0"): _cot("m", "x0", "matematika", 0),
         ("m", "gone"): _cot("m", "gone", "ona_tili", 1)}   # absent from session a
    assert compare(a, b, subject="ona_tili")["m"] == {
        "n": 2, "churn": 1, "churn_pct": 50.0,
        "acc_a": 100.0, "acc_b": 50.0, "net_drift": -50.0}
    assert compare(a, b)["m"]["n"] == 3      # both subjects, still only shared items
