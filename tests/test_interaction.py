import json
import pytest
from analysis.interaction import analyse, cell_contributions, load_dir, strata_from
from src.stats import fisher_exact_2x2, mantel_haenszel_or

NATIVE = "nat"


def _item(model, cond, qid, is_correct, subject=NATIVE):
    return {"model": model, "condition": cond, "subject": subject, "qid": qid,
            "correct": "A", "is_correct": is_correct, "rescue_correct": is_correct,
            "predicted": "A", "parse_error": False, "truncated": False}


def test_fisher_exact_matches_hand_computed():
    # Fisher's tea tasting: [[3,1],[1,3]] -> 0.4857
    assert fisher_exact_2x2(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    # perfect separation, 10 vs 10
    assert fisher_exact_2x2(10, 0, 0, 10) == pytest.approx(2 / 184756, abs=1e-9)
    # no association -> 1.0
    assert fisher_exact_2x2(5, 5, 5, 5) == pytest.approx(1.0)
    # degenerate margins are uninformative, not an error
    assert fisher_exact_2x2(0, 0, 4, 4) == 1.0
    assert fisher_exact_2x2(0, 0, 0, 0) == 1.0


def test_mantel_haenszel_reduces_to_plain_or_on_one_stratum():
    assert mantel_haenszel_or([(8, 3, 8, 12)]) == pytest.approx(8 * 12 / (3 * 8))
    assert mantel_haenszel_or([]) == float("inf")
    assert mantel_haenszel_or([(1, 0, 0, 1)]) == float("inf")   # zero denominator


def test_mantel_haenszel_pools_two_strata():
    # stratum 1 (n=4):  num += 1*1/4 = 0.25      den += 1*1/4 = 0.25
    # stratum 2 (n=12): num += 4*4/12 = 1.3333   den += 2*2/12 = 0.3333
    strata = [(1, 1, 1, 1), (4, 2, 2, 4)]
    assert mantel_haenszel_or(strata) == pytest.approx((0.25 + 16 / 12) / (0.25 + 4 / 12))


def test_mantel_haenszel_is_not_the_pooled_table():
    """Stratification must resist Simpson's paradox: each stratum shows OR=1 but the
    naively collapsed table does not."""
    strata = [(9, 1, 9, 1), (1, 9, 1, 9)]
    assert mantel_haenszel_or(strata) == pytest.approx(1.0)


def test_analyse_counts_and_odds_ratio(tmp_path):
    # one model; native: 3 lost / 1 gained, other: 1 lost / 3 gained -> OR = 9
    items = []
    for i, (a, b) in enumerate([(1, 0), (1, 0), (1, 0), (0, 1)]):
        items += [_item("m", "cot", f"n{i}", a), _item("m", "dspy_bootstrap", f"n{i}", b)]
    for i, (a, b) in enumerate([(1, 0), (0, 1), (0, 1), (0, 1)]):
        items += [_item("m", "cot", f"o{i}", a, subject="other"),
                  _item("m", "dspy_bootstrap", f"o{i}", b, subject="other")]
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))

    out = analyse(str(tmp_path), NATIVE, n_boot=200)
    r = out["per_model"]["m"]
    assert (r["native_lost"], r["native_gained"]) == (3, 1)
    assert (r["other_lost"], r["other_gained"]) == (1, 3)
    assert r["odds_ratio"] == pytest.approx(9.0)
    assert r["native_harm_share"] == 75.0 and r["other_harm_share"] == 25.0
    assert out["mh_odds_ratio"] == pytest.approx(9.0)
    assert out["models_agreeing"] == 1 and out["models_comparable"] == 1
    assert out["preregistered"] is False
    assert out["n_items"] == 8


def test_concordant_items_are_excluded_but_still_clustered(tmp_path):
    """Items that do not flip contribute nothing to the tables, yet must still be part
    of the item universe the bootstrap resamples."""
    items = [_item("m", "cot", "n0", 1), _item("m", "dspy_bootstrap", "n0", 1),
             _item("m", "cot", "n1", 1), _item("m", "dspy_bootstrap", "n1", 0),
             _item("m", "cot", "o0", 1, subject="other"),
             _item("m", "dspy_bootstrap", "o0", 0, subject="other")]
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    rows = cell_contributions(load_dir(str(tmp_path)), NATIVE)
    assert set(rows) == {"n0", "n1", "o0"}                 # concordant item retained
    assert rows["n0"] == [("m", True, 0, 0)]               # ...contributing zeros
    assert strata_from(rows, ["n0"])["m"] == [0, 0, 0, 0]


def _write(tmp_path, per_model):
    """One *_items.jsonl per model, as the runners emit."""
    for model, items in per_model.items():
        (tmp_path / f"{model}_items.jsonl").write_text(
            "\n".join(json.dumps(i) for i in items))


def _pairs(model, prefix, subject, n_lost, n_gained):
    """n_lost items that cot got right and bootstrap got wrong, and vice versa."""
    items = []
    for i in range(n_lost):
        q = f"{prefix}L{i}"
        items += [_item(model, "cot", q, 1, subject),
                  _item(model, "dspy_bootstrap", q, 0, subject)]
    for i in range(n_gained):
        q = f"{prefix}G{i}"
        items += [_item(model, "cot", q, 0, subject),
                  _item(model, "dspy_bootstrap", q, 1, subject)]
    return items


def test_bootstrap_is_deterministic_under_seed(tmp_path):
    # all four cells non-zero in every stratum, so the odds ratio is identified
    _write(tmp_path, {m: _pairs(m, "n", NATIVE, 3, 2) + _pairs(m, "o", "other", 2, 3)
                      for m in ("m1", "m2")})
    a = analyse(str(tmp_path), NATIVE, n_boot=300, seed=7)
    b = analyse(str(tmp_path), NATIVE, n_boot=300, seed=7)
    assert a["bootstrap"]["ci95"] == b["bootstrap"]["ci95"] is not None
    assert a["bootstrap"]["seed"] == 7
    assert analyse(str(tmp_path), NATIVE, n_boot=300, seed=8)["bootstrap"]["n"] > 0


def test_bootstrap_key_present_when_odds_ratio_unidentified(tmp_path):
    """No non-native discordance -> every draw is inf -> report it, don't KeyError."""
    _write(tmp_path, {"m": [_item("m", "cot", "n0", 1),
                            _item("m", "dspy_bootstrap", "n0", 0),
                            _item("m", "cot", "o0", 1, subject="other"),
                            _item("m", "dspy_bootstrap", "o0", 1, subject="other")]})
    out = analyse(str(tmp_path), NATIVE, n_boot=50)
    assert out["bootstrap"]["ci95"] is None and out["bootstrap"]["n"] == 0
    assert out["mh_odds_ratio"] == float("inf")


def test_items_shared_across_models_resample_together(tmp_path):
    """The clustering guarantee: drawing an item pulls in every model's response to it,
    so repeated items cannot be treated as independent observations."""
    _write(tmp_path, {m: [_item(m, "cot", "n0", 1),
                          _item(m, "dspy_bootstrap", "n0", 0)]
                      for m in ("m1", "m2", "m3")})
    rows = cell_contributions(load_dir(str(tmp_path)), NATIVE)
    assert len(rows["n0"]) == 3                            # all three models, one cluster
    strata = strata_from(rows, ["n0", "n0"])               # item drawn twice
    assert len(strata) == 3
    assert all(s == [2, 0, 0, 0] for s in strata.values())  # each model doubles


def test_load_dir_rejects_mixed_and_duplicate_rows(tmp_path):
    (tmp_path / "bad_items.jsonl").write_text("\n".join(json.dumps(i) for i in [
        _item("m1", "cot", "n0", 1), _item("m2", "cot", "n0", 1)]))
    with pytest.raises(SystemExit, match="one model per file"):
        load_dir(str(tmp_path))
    (tmp_path / "bad_items.jsonl").write_text("\n".join(json.dumps(i) for i in [
        _item("m1", "cot", "n0", 1), _item("m1", "cot", "n0", 0)]))
    with pytest.raises(SystemExit, match="duplicate"):
        load_dir(str(tmp_path))


def _trace(cond, subject, gold, is_correct):
    return {"condition": cond, "subject": subject, "correct": gold,
            "is_correct": is_correct, "predicted": gold, "reasoning": ""}


def test_traces_loader_uses_positional_qids(tmp_path):
    rows = []
    for cond in ("cot", "dspy_bootstrap"):
        rows += [_trace(cond, NATIVE, "A", 1), _trace(cond, "other", "B", 0)]
    (tmp_path / "m1_traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    data = load_dir(str(tmp_path))
    assert set(data) == {"m1"}
    assert {i["qid"] for i in data["m1"]} == {"pos0000", "pos0001"}
    assert all(i["truncated"] is False and i["parse_error"] is False
               for i in data["m1"])


def test_traces_loader_rejects_misaligned_conditions(tmp_path):
    rows = [_trace("cot", NATIVE, "A", 1), _trace("cot", "other", "B", 1),
            _trace("dspy_bootstrap", "other", "B", 1),      # swapped order
            _trace("dspy_bootstrap", NATIVE, "A", 1)]
    (tmp_path / "m1_traces.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(SystemExit, match="item order differs"):
        load_dir(str(tmp_path))


def test_traces_loader_rejects_models_with_different_item_sequences(tmp_path):
    a = [_trace(c, NATIVE, "A", 1) for c in ("cot", "dspy_bootstrap")]
    b = [_trace(c, NATIVE, "D", 1) for c in ("cot", "dspy_bootstrap")]   # other gold
    (tmp_path / "m1_traces.jsonl").write_text("\n".join(json.dumps(r) for r in a))
    (tmp_path / "m2_traces.jsonl").write_text("\n".join(json.dumps(r) for r in b))
    with pytest.raises(SystemExit, match="sequence differs"):
        load_dir(str(tmp_path))


def test_original_stack_reproduces_published_discordant_counts():
    """Regression pin: the traces loader must reproduce the b=75, c=46 that back the
    published pooled McNemar p=0.011 on ona_tili. A pairing bug would move these."""
    import os
    if not os.path.isdir("results/main"):
        pytest.skip("original-stack traces not present")
    out = analyse("results/main", "ona_tili", n_boot=1)
    assert out["pooled"]["native_lost"] == 75
    assert out["pooled"]["native_gained"] == 46
    assert out["n_models"] == 6


def test_contrast_is_parameterisable(tmp_path):
    """E8 scores cot -> dspy_mipro through the same estimator as cot -> bootstrap."""
    items = (_pairs("m", "n", NATIVE, 3, 1) + _pairs("m", "o", "other", 1, 3))
    for i in items:
        if i["condition"] == "dspy_bootstrap":
            i["condition"] = "dspy_mipro"
    _write(tmp_path, {"m": items})
    out = analyse(str(tmp_path), NATIVE, n_boot=50, cond_b="dspy_mipro")
    assert out["contrast"] == "cot -> dspy_mipro"
    assert out["mh_odds_ratio"] == pytest.approx(9.0)


def test_missing_condition_fails_loudly(tmp_path):
    """A typo'd or absent condition must error, not silently yield an empty table."""
    _write(tmp_path, {"m": _pairs("m", "n", NATIVE, 2, 1)})
    with pytest.raises(SystemExit, match="missing condition"):
        analyse(str(tmp_path), NATIVE, n_boot=10, cond_b="dspy_nonexistent")


def test_zero_discordance_model_flagged_as_uninformative(tmp_path):
    """A model the optimizer no-oped on contributes no discordant pairs. That is a
    structural null, not evidence of no harm, and must be reported as such."""
    _write(tmp_path, {
        "real": _pairs("real", "n", NATIVE, 3, 1) + _pairs("real", "o", "other", 1, 3),
        # every item concordant: cot and the optimized program agree everywhere
        "noop": [_item("noop", c, f"n{i}", 1)
                 for c in ("cot", "dspy_bootstrap") for i in range(4)]
                + [_item("noop", c, f"o{i}", 1, subject="other")
                   for c in ("cot", "dspy_bootstrap") for i in range(4)],
    })
    out = analyse(str(tmp_path), NATIVE, n_boot=100)
    assert out["n_models"] == 2
    assert out["n_informative"] == 1
    assert out["uninformative_models"] == ["noop"]
    assert out["per_model"]["noop"]["discordant_pairs"] == 0
    assert out["per_model"]["real"]["discordant_pairs"] == 8
    # the no-op must not move the pooled estimate
    assert out["mh_odds_ratio"] == pytest.approx(9.0)


# --- subject-TYPE grouping (knowledge vs reasoning) -------------------------------

def _gitem(cond, qid, subject, is_correct):
    return {"model": "m", "condition": cond, "subject": subject, "qid": qid,
            "correct": "A", "is_correct": is_correct, "rescue_correct": is_correct,
            "predicted": "A", "parse_error": False, "truncated": False}


def test_group_label_pools_subjects_into_one_arm(tmp_path):
    """Passing a collection makes the contrast group-vs-rest. Two knowledge subjects
    each losing 4 and gaining 1 must pool to 8/2 against the reasoning arm -- i.e. the
    grouped run is the same estimator with a coarser label, not a different one."""
    rows = []
    for subj in ("ona_tili", "tarix"):            # 4 lost, 1 gained each -> 8/2
        for q in range(5):
            rows += [_gitem("cot", f"{subj}{q}", subj, 1 if q < 4 else 0),
                     _gitem("dspy_bootstrap", f"{subj}{q}", subj, 0 if q < 4 else 1)]
    for q in range(10):                           # reasoning: 2 lost, 8 gained
        rows += [_gitem("cot", f"r{q}", "matematika", 1 if q < 2 else 0),
                 _gitem("dspy_bootstrap", f"r{q}", "matematika", 0 if q < 2 else 1)]
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    g = analyse(str(tmp_path), ["ona_tili", "tarix"], n_boot=200)
    assert g["pooled"]["native_lost"] == 8 and g["pooled"]["native_gained"] == 2
    assert g["pooled"]["other_lost"] == 2 and g["pooled"]["other_gained"] == 8
    assert g["mh_odds_ratio"] == 16.0             # (8*8)/(2*2)
    # the label is recorded as a sorted list, so a grouped run says what it pooled
    assert g["native"] == ["ona_tili", "tarix"]


def test_single_subject_string_behaviour_is_unchanged(tmp_path):
    """A one-element group and the bare string must agree exactly: the generalisation
    must not perturb any single-subject number already printed in the paper."""
    rows = []
    for q in range(6):
        rows += [_gitem("cot", f"n{q}", "ona_tili", 1 if q < 4 else 0),
                 _gitem("dspy_bootstrap", f"n{q}", "ona_tili", 0 if q < 4 else 1)]
    for q in range(6):
        rows += [_gitem("cot", f"o{q}", "tarix", 1 if q < 2 else 0),
                 _gitem("dspy_bootstrap", f"o{q}", "tarix", 0 if q < 2 else 1)]
    (tmp_path / "m_items.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    as_str = analyse(str(tmp_path), "ona_tili", n_boot=200)
    as_list = analyse(str(tmp_path), ["ona_tili"], n_boot=200)
    assert as_str["mh_odds_ratio"] == as_list["mh_odds_ratio"]
    assert as_str["pooled"] == as_list["pooled"]
