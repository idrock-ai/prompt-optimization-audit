#!/usr/bin/env python3
"""Specificity (placebo) check on the differential-harm estimator.

The differential in analysis/interaction.py asks: conditional on an item changing
correctness, are the odds the change is harmful higher for the native subject than for
the rest? The obvious objection is that "native subject" is a proxy for something else
-- a low-accuracy subject, or a knowledge subject as opposed to a reasoning one -- and
that the estimator would flag whichever subject the optimizer happens to help least.

This script answers that by rotating the `native` label through EVERY subject in a
benchmark and re-running the identical estimator. If the differential is specific to the
native language, only the true native subject should depart from 1; in particular the
benchmark's other knowledge subject (DTM `tarix`, TurkishMMLU `History`) -- the closest
structural analogue to a language-and-literature subject -- should sit at 1.

It also reports each subject's CoT accuracy, because the competing explanation runs
through baseline accuracy: under a noise model in which the optimizer perturbs answers
at random, a subject with more correct answers has more to lose, so HIGH-accuracy
subjects should show the higher harm-odds. Printing accuracy alongside the odds ratio
lets a reader check the direction of that confound rather than take our word for it.

Usage: python analysis/placebo.py results/main --native ona_tili
"""
import argparse, collections, json, sys
sys.path.insert(0, ".")
from analysis.interaction import analyse, load_dir


def subject_accuracy(data, cond="cot"):
    """{subject: (pct_correct, n)} pooled over models, in the reference condition."""
    acc = collections.defaultdict(lambda: [0, 0])
    for items in data.values():
        for i in items:
            if i["condition"] == cond:
                a = acc[i["subject"]]
                a[0] += bool(i["is_correct"])
                a[1] += 1
    return {s: (100 * k / n, n) for s, (k, n) in acc.items() if n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/main")
    ap.add_argument("--native", default="ona_tili", help="the true native subject")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--cond-a", default="cot")
    ap.add_argument("--cond-b", default="dspy_bootstrap")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    data = load_dir(a.dir)
    if not data:
        sys.exit(f"no per-item logs in {a.dir}")
    acc = subject_accuracy(data, a.cond_a)
    subjects = sorted(acc, key=lambda s: -acc[s][0])

    rows = []
    for s in subjects:
        r = analyse(a.dir, s, n_boot=a.boot, cond_a=a.cond_a, cond_b=a.cond_b)
        ci = r["bootstrap"]["ci95"]
        rows.append({"subject": s, "is_native": s == a.native,
                     "cot_accuracy": round(acc[s][0], 1), "n_items": acc[s][1],
                     "mh_odds_ratio": r["mh_odds_ratio"], "ci95": ci,
                     "models_agreeing": r["models_agreeing"],
                     "models_comparable": r["models_comparable"]})

    print(f"=== placebo rotation, {a.dir} ({a.cond_a} -> {a.cond_b}) [NOT pre-registered]")
    print(f"{'subject':34} {'CoT acc':>8} {'OR':>6} {'95% CI':>16} {'agree':>7}")
    for r in rows:
        ci = "[{:.2f}, {:.2f}]".format(*r["ci95"]) if r["ci95"] else "n/a"
        mark = " *native*" if r["is_native"] else ""
        print(f"{r['subject']:34} {r['cot_accuracy']:7.1f}% {r['mh_odds_ratio']:6.2f} "
              f"{ci:>16} {r['models_agreeing']}/{r['models_comparable']:<5}{mark}")

    nat = next((r for r in rows if r["is_native"]), None)
    others = [r for r in rows if not r["is_native"]]
    if nat and others:
        worst = max(others, key=lambda r: r["mh_odds_ratio"])
        print(f"\nnative OR {nat['mh_odds_ratio']} vs. largest non-native "
              f"{worst['mh_odds_ratio']} ({worst['subject']})")
        # Direction of the baseline-accuracy confound: a pure-noise perturbation model
        # predicts HIGHER harm-odds for HIGHER-accuracy subjects (more to lose).
        higher = [r for r in others if r["cot_accuracy"] > nat["cot_accuracy"]]
        if higher and all(r["mh_odds_ratio"] < nat["mh_odds_ratio"] for r in higher):
            print(f"baseline-accuracy confound runs the OPPOSITE way: all {len(higher)} "
                  f"subjects with higher CoT accuracy than {a.native} have LOWER odds.")

    dest = a.out or f"{a.dir}/placebo.json"
    json.dump({"dir": a.dir, "native": a.native, "contrast": f"{a.cond_a} -> {a.cond_b}",
               "preregistered": False, "rows": rows}, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
