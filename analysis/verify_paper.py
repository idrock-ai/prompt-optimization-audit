#!/usr/bin/env python3
"""Assert that the numbers printed in paper/main.tex match the committed logs.

Every claim below is stated as (description, value in the paper, value recomputed from
results/). The script recomputes wherever it cheaply can rather than reading a cached
export, so a stale results/paper_numbers.json cannot make a wrong paper look right.

Run before every submission or arXiv upload:  python analysis/verify_paper.py
Exit status is non-zero if any check fails or if a paper number has drifted.
"""
import glob, json, math, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact
from analysis.decompose import decompose_dir
from analysis.dose_response import collect
from analysis.determinism import load_cot, compare
from analysis.interaction import analyse

FAILURES = []
KNOWLEDGE = {"ona_tili", "tarix"}
REASONING = {"fizika", "matematika"}


def roundn(x, n):
    """n decimals, ties away from zero -- see round1 for why this exists."""
    m = 10 ** n
    return math.floor(abs(x) * m + 0.5) / m * (1 if x >= 0 else -1)


def round1(x):
    """One decimal, ties away from zero.

    Several arm means land on an exact .25 (four models scoring in whole percentage
    points), where Python's round() goes half-to-even and returns 33.2 for 33.25 while
    the paper prints the conventional 33.3. Encoding the convention here keeps the
    verifier honest instead of loosening the tolerance until the mismatch disappears."""
    return math.floor(abs(x) * 10 + 0.5) / 10 * (1 if x >= 0 else -1)


def check(desc, in_paper, computed, tol=0.05):
    ok = (abs(in_paper - computed) <= tol if isinstance(computed, float)
          else in_paper == computed)
    print(f"{'ok  ' if ok else 'FAIL'}  {desc:54} paper={in_paper}  computed={computed}")
    if not ok:
        FAILURES.append(desc)


def e9_pairs():
    """[(subject, cot_correct, boot_correct)] over every model in the powered run."""
    out = []
    for f in sorted(glob.glob("results/e9/*_items.jsonl")):
        A, B = {}, {}
        for line in open(f):
            r = json.loads(line)
            key = (r["model"], r["qid"])
            if r["condition"] == "cot":
                A[key] = r
            elif r["condition"] == "dspy_bootstrap":
                B[key] = r
        for k, a in A.items():
            b = B.get(k)
            if b:
                out.append((a["subject"], bool(a["is_correct"]), bool(b["is_correct"])))
    return out


def group_stats(pairs, subjects=None):
    sel = [p for p in pairs if subjects is None or p[0] in subjects]
    b = sum(1 for _, x, y in sel if x and not y)
    c = sum(1 for _, x, y in sel if not x and y)
    cot = 100 * sum(1 for _, x, _ in sel if x) / len(sel)
    boot = 100 * sum(1 for _, _, y in sel if y) / len(sel)
    return {"b": b, "c": c, "n": len(sel), "p": mcnemar_exact(b, c),
            "cot": cot, "boot": boot, "delta": boot - cot}


def main():
    pairs = e9_pairs()
    if not pairs:
        sys.exit("no results/e9 logs -- the powered audit has not been run")

    print("== Sec. 4: the powered audit (Table 1)")
    for label, subs, d_paper, p_paper, n_paper in (
            ("native (ona_tili)", {"ona_tili"}, -1.1, 0.283, 2358),
            ("other-knowledge (tarix)", {"tarix"}, -1.4, 0.344, 900),
            ("knowledge (both)", KNOWLEDGE, -1.2, 0.150, 3258),
            ("reasoning", REASONING, +2.3, 0.029, 1800),
            ("overall", None, +0.1, 0.925, 5058)):
        s = group_stats(pairs, subs)
        check(f"{label}: n", n_paper, s["n"])
        check(f"{label}: delta", d_paper, round(s["delta"], 1), 0.05)
        check(f"{label}: p", p_paper, round(s["p"], 3), 0.001)
    for label, subs, cot_paper in (("native", {"ona_tili"}, 35.8),
                                   ("tarix", {"tarix"}, 55.9),
                                   ("reasoning", REASONING, 69.7),
                                   ("overall", None, 51.5)):
        check(f"{label}: CoT accuracy", cot_paper,
              round(group_stats(pairs, subs)["cot"], 1), 0.05)

    print("\n== Sec. 4: the placebo rotation (Table 2)")
    small = {r["subject"]: r for r in json.load(open("results/main/placebo.json"))["rows"]}
    powered = {r["subject"]: r for r in json.load(open("results/e9/placebo.json"))["rows"]}
    for subj, s_or, p_or in (("ona_tili", 2.27, 1.25), ("tarix", 1.04, 1.16),
                             ("fizika", 0.44, 0.78), ("matematika", 0.47, 0.75)):
        check(f"{subj}: OR at n~100", s_or, round(small[subj]["mh_odds_ratio"], 2), 0.01)
        check(f"{subj}: OR powered", p_or, round(powered[subj]["mh_odds_ratio"], 2), 0.01)
    gap_small = small["ona_tili"]["mh_odds_ratio"] - small["tarix"]["mh_odds_ratio"]
    gap_pow = powered["ona_tili"]["mh_odds_ratio"] - powered["tarix"]["mh_odds_ratio"]
    check("native-history gap at n~100", 1.23, round(gap_small, 2), 0.01)
    check("native-history gap powered", 0.09, round(gap_pow, 2), 0.01)
    check("powered: native CI includes 1", True,
          powered["ona_tili"]["ci95"][0] < 1.0)

    print("\n== Sec. 4: knowledge vs reasoning, tested directly (Table 2, last row)")
    # Recomputed from the logs, not read from a cached export: this is the paper's
    # central claim as a single estimate, so it is the last number that should be
    # allowed to drift silently.
    grp_small = analyse("results/main", sorted(KNOWLEDGE), n_boot=4000)
    grp_pow = analyse("results/e9", sorted(KNOWLEDGE), n_boot=4000)
    check("knowledge-vs-reasoning OR at n~100", 2.67,
          round(grp_small["mh_odds_ratio"], 2), 0.01)
    check("knowledge-vs-reasoning OR powered", 1.40,
          round(grp_pow["mh_odds_ratio"], 2), 0.01)
    check("powered CI low", 1.08, round(grp_pow["bootstrap"]["ci95"][0], 2), 0.01)
    check("powered CI high", 1.81, round(grp_pow["bootstrap"]["ci95"][1], 2), 0.01)
    check("powered bootstrap P(OR<=1)", 0.008,
          round(grp_pow["bootstrap"]["p_or_le_1"], 3), 0.0005)
    # the claim that distinguishes this contrast from the native one: it SURVIVES
    check("powered: subject-type CI excludes 1", True,
          grp_pow["bootstrap"]["ci95"][0] > 1.0)
    check("attenuates under powering (2.67 > 1.40)", True,
          grp_small["mh_odds_ratio"] > grp_pow["mh_odds_ratio"])
    # heterogeneity the paper states rather than hides: 4 of 6 models above 1, both
    # gemma models below
    ors = {m: r["odds_ratio"] for m, r in grp_pow["per_model"].items()}
    check("models with OR > 1", 4, sum(1 for v in ors.values() if v and v > 1))
    check("both gemma models invert", True,
          all(v < 1 for m, v in ors.items() if "gemma" in m and v))

    print("\n== Sec. 4: the models do not agree (Table 2, lower rows)")
    hp, hs = grp_pow["heterogeneity"], grp_small["heterogeneity"]
    check("powered Cochran Q", 14.9, round(hp["Q"], 1), 0.05)
    check("powered Q df", 5, hp["df"])
    check("powered Q p", 0.011, round(hp["p"], 3), 0.0005)
    check("powered I2 (%)", 66, round(hp["I2"]), 0.5)
    check("powered: common-OR assumption REJECTED", True, hp["p"] < 0.05)
    rp = grp_pow["random_effects"]
    check("powered random-effects OR", 1.33, rp["or"], 0.005)
    check("RE analytic CI (models exchangeable) spans 1", True,
          rp["ci95_analytic"][0] < 1.0 < rp["ci95_analytic"][1])
    check("RE analytic CI low", 0.83, round(rp["ci95_analytic"][0], 2), 0.005)
    check("RE analytic CI high", 2.14, round(rp["ci95_analytic"][1], 2), 0.005)
    check("RE cluster-boot CI low", 1.01, round(rp["ci95_cluster_boot"][0], 2), 0.005)
    check("RE cluster-boot CI high", 1.73, round(rp["ci95_cluster_boot"][1], 2), 0.005)
    # 86/4000 = 0.0215 exactly; Python's half-to-even would print 0.021
    check("RE cluster-boot P(OR<=1)", 0.022,
          roundn(rp["p_or_le_1_cluster_boot"], 3), 0.0005)
    # the second-order claim: powering EXPOSED the disagreement
    check("n~100 Cochran Q", 8.5, round(hs["Q"], 1), 0.05)
    check("n~100 Q p", 0.13, round(hs["p"], 2), 0.005)
    check("n~100 I2 (%)", 41, round(hs["I2"]), 0.5)
    check("I2 ROSE under powering (41 -> 66)", True, hp["I2"] > hs["I2"])
    check("n~100 random-effects OR", 2.53, round(grp_small["random_effects"]["or"], 2), 0.005)
    nat_small = analyse("results/main", "ona_tili", n_boot=200)["heterogeneity"]
    check("n~100 native I2 (%)", 20, round(nat_small["I2"]), 0.5)
    check("n~100 native Q p", 0.28, round(nat_small["p"], 2), 0.005)

    print("\n== Sec. 4: null on every scoring")
    e9 = decompose_dir("results/e9")["pooled"]
    check("deployment p", 0.283, e9["deployment_p"], 0.001)
    check("knowledge p", 0.602, e9["knowledge_p"], 0.001)
    check("knowledge direction reversed (c>b)", True,
          e9["knowledge"]["c"] > e9["knowledge"]["b"])
    check("rescue p", 0.755, e9["rescue_p"], 0.001)

    print("\n== Sec. 5: substitution ablation (Table 3)")
    sub = json.load(open("results/e10/substitution_stats.json"))
    check("constraint p (null)", 0.900, sub["pooled"]["constraint"]["p"], 0.001)
    check("substitution p", 0.0097, sub["pooled"]["substitution"]["p"], 0.0005)
    check("constraint b/c", [126, 129],
          [sub["pooled"]["constraint"]["b"], sub["pooled"]["constraint"]["c"]])
    check("substitution b/c", [105, 147],
          [sub["pooled"]["substitution"]["b"], sub["pooled"]["substitution"]["c"]])
    means = {k: sum(r[k] for r in sub["per_model"].values()) / len(sub["per_model"])
             for k in ("cot", "vanilla", "compliant",
                       "random_compliant", "random_noncompliant")}
    check("mean CoT", 33.3, round1(means["cot"]), 0.05)
    check("mean vanilla", 28.5, round1(means["vanilla"]), 0.05)
    check("mean compliant", 31.3, round1(means["compliant"]), 0.05)
    check("mean random-compliant", 32.3, round1(means["random_compliant"]), 0.05)
    check("mean random-noncompliant", 32.0, round1(means["random_noncompliant"]), 0.05)
    check("vanilla is the WORST arm", True,
          means["vanilla"] < min(v for k, v in means.items() if k != "vanilla"))
    check("random demos beat vanilla by (pts)", 3.5,
          round1(means["random_noncompliant"] - means["vanilla"]), 0.05)

    print("\n== Sec. 6: format route -- causal, and a minority")
    dose = collect("results/e2", e1_dir="results/e1")
    treated = ["gemma4:e4b", "qwen3.5:9b"]
    tb = sum(dose[m]["outcome"]["b_lo_right_hi_wrong"] for m in treated)
    tc = sum(dose[m]["outcome"]["c_lo_wrong_hi_right"] for m in treated)
    check("budget outcome b/c", [11, 25], [tb, tc])
    check("budget outcome p", 0.029, round(mcnemar_exact(tb, tc), 3), 0.001)
    ctl = dose["gemma4:31b"]["outcome"]
    check("zero-truncation control p", 0.69,
          round(mcnemar_exact(ctl["b_lo_right_hi_wrong"],
                              ctl["c_lo_wrong_hi_right"]), 2), 0.01)
    check("format share of harmful flips (%)", 17.6, e9["format_share_pct"], 0.1)
    check("format-attributable flips", 46, e9["format_flips"])
    check("harmful flips total", 262, e9["harmful_flips"])
    check("no format-drift flips", 0, e9["flip_classes"]["format_drift"])

    print("\n== Sec. 5/6: what BootstrapFewShot actually selected")
    # Rebuild the exact pool the optimizer saw: capped per subject, then shuffled with
    # the run seed. The subject mix of THIS pool is the correct baseline -- comparing
    # against the benchmark's mix is what produced the retracted 2.3x figure.
    from src.data import load_splits, cap_per_subject
    import random as _random
    train, _dev, _test = load_splits(seed=42)
    pool = cap_per_subject(train, 40)
    _random.Random(42).shuffle(pool)
    qsub = {ex.question[:45]: ex.subject for ex in pool}
    slots, distinct, per_model = [], {}, {}
    for f in sorted(glob.glob("results/e1/*_bootstrap.json")):
        demos = [d["question"][:45] for d in json.load(open(f))["predict"]["demos"]]
        per_model[f] = set(demos)
        slots += demos
        for q in demos:
            distinct[q] = None
    reasoning = {"matematika", "fizika"}
    pct = lambda ks: 100 * sum(1 for k in ks if qsub[k] in reasoning) / len(ks)
    # Measure the pool from the examples themselves, not from `qsub`: that dict is keyed
    # on a 45-character question prefix, and two pool items share one, so keying through
    # it silently drops an example and reports 49.7%.
    check("pool the optimizer saw: % reasoning", 50.0,
          round(100 * sum(1 for ex in pool if ex.subject in reasoning) / len(pool), 1),
          0.05)
    check("selected, over slots: % reasoning", 58.0, round(pct(slots), 1), 0.5)
    check("selected, over distinct questions: % reasoning", 50.0,
          round(pct(list(distinct)), 1), 0.5)
    check("demonstration slots across six models", 24, len(slots))
    check("DISTINCT questions selected", 8, len(distinct))
    sets = list(per_model.values())
    identical = sum(1 for i in range(len(sets)) for j in range(i + 1, len(sets))
                    if sets[i] == sets[j])
    check("model pairs with identical demo sets", 1, identical)

    print("\n== Limitations: determinism")
    det = compare(load_cot("results/e1"), load_cot("results/e8"), subject="ona_tili")
    check("gemma4:e4b ona churn", 12, det["gemma4:e4b"]["churn"])
    check("gemma4:31b ona churn", 7, det["gemma4:31b"]["churn"])
    check("gemma4:e4b net drift", -2.0, det["gemma4:e4b"]["net_drift"], 0.01)
    check("gemma4:31b net drift", 1.0, det["gemma4:31b"]["net_drift"], 0.01)
    check("all four qwen models reproduce exactly", 4,
          sum(1 for m, v in det.items() if m.startswith("qwen") and v["churn"] == 0))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("all paper numbers verified against committed logs")


if __name__ == "__main__":
    main()
