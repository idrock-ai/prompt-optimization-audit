import json

import dspy

from src.substitution import draw, make_demo, split_by_compliance
from analysis.substitution_stats import cell_arms, cell_mean, acc, paired


def _ex(q, subject="ona_tili", letter="A"):
    e = dspy.Example(question=q, options="A) x\nB) y", answer_letter=letter,
                     subject=subject).with_inputs("question", "options")
    e.subject = subject
    return e


def test_split_by_compliance_uses_the_prompts_own_rule():
    short = make_demo(_ex("q1"), "Qisqa javob.")
    long = make_demo(_ex("q2"), "Bir. Ikki. Uch. To'rt. " + "x" * 400)
    compliant, noncompliant = split_by_compliance([short, long])
    assert compliant == [short] and noncompliant == [long]


def test_draw_is_deterministic_and_seed_dependent():
    cands = [make_demo(_ex(f"q{i}"), "Qisqa.") for i in range(20)]
    a = [d.question for d in draw(cands, 4, seed=1)]
    b = [d.question for d in draw(cands, 4, seed=1)]
    c = [d.question for d in draw(cands, 4, seed=2)]
    assert a == b and len(a) == 4
    assert a != c


def test_draw_is_without_replacement_and_clamps_to_pool_size():
    cands = [make_demo(_ex(f"q{i}"), "Qisqa.") for i in range(3)]
    got = draw(cands, 4, seed=1)
    assert len(got) == 3
    assert len({d.question for d in got}) == 3


def _item(model, cond, qid, is_correct, subject="ona_tili"):
    return {"model": model, "condition": cond, "subject": subject, "qid": qid,
            "correct": "A", "is_correct": is_correct, "rescue_correct": is_correct,
            "predicted": "A", "parse_error": False, "truncated": False}


def test_cell_arms_and_mean_average_over_seeds():
    """A single four-demo draw is noisy; the contrast is defined on the seed mean."""
    items = []
    for s, correct in enumerate((1, 1, 0)):          # 100%, 100%, 0% -> mean 66.7
        items += [_item("m", f"random_compliant_s{s}", "q1", correct)]
    items += [_item("m", "random_noncompliant_s0", "q1", 0)]
    assert cell_arms(items, "random_compliant") == [
        "random_compliant_s0", "random_compliant_s1", "random_compliant_s2"]
    assert abs(cell_mean(items, "random_compliant", "ona_tili") - 200 / 3) < 1e-9
    # the noncompliant prefix must not swallow the compliant arms
    assert cell_arms(items, "random_noncompliant") == ["random_noncompliant_s0"]


def test_paired_only_uses_shared_items():
    items = [_item("m", "a", "q1", 1), _item("m", "a", "q2", 1),
             _item("m", "b", "q1", 0), _item("m", "b", "q3", 1)]
    assert paired(items, "a", "b", "ona_tili") == (1, 0)


def test_acc_is_subject_scoped():
    items = [_item("m", "a", "q1", 1, "ona_tili"), _item("m", "a", "q2", 0, "ona_tili"),
             _item("m", "a", "q3", 1, "matematika")]
    assert acc(items, "a", "ona_tili") == 50.0
    assert acc(items, "a", "matematika") == 100.0
    assert acc(items, "a", "fizika") is None


def test_demo_field_reads_both_example_and_dict_forms():
    """Module.load() restores demos as plain dicts. getattr on a dict silently returns
    the default, which reported every saved demonstration as zero-length -- and so as
    trivially brevity-compliant. Both forms must read the same."""
    from src.program import demo_field, demo_payload
    ex = make_demo(_ex("q"), "Uzun mulohaza. Ikkinchi gap. Uchinchi gap. " + "x" * 400)
    as_dict = {"question": ex.question, "options": ex.options,
               "reasoning": ex.reasoning, "answer_letter": "A"}
    assert demo_field(ex, "reasoning") == demo_field(as_dict, "reasoning")
    assert demo_field(as_dict, "missing", "fallback") == "fallback"

    for demos in ([ex], [as_dict]):
        p = demo_payload(demos)
        assert p["n_demos"] == 1
        assert p["reasoning_total"] > 400          # not silently zero
        assert p["payload_total"] > p["reasoning_total"]
        assert p["n_compliant"] == 0               # long trace is not compliant


def test_demo_payload_on_a_real_saved_program():
    """Guards the exact regression: results/e1 programs must not report 0 chars."""
    import pathlib
    from src.program import CoTSolver, demo_payload
    path = pathlib.Path("results/e1/gemma4_e4b_bootstrap.json")
    if not path.is_file():
        return
    prog = CoTSolver(); prog.load(str(path))
    p = demo_payload(prog.predict.demos)
    assert p["n_demos"] == 4
    assert p["reasoning_total"] > 2000
