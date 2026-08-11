#!/usr/bin/env python3
"""E10: does the compliant metric work through CONSTRAINT or through SUBSTITUTION?

Averages the random draws within each compliance cell (a single four-demo draw is
noisy), then reports the two contrasts that decide it:

  constraint effect   random_compliant  -  random_noncompliant
      Both cells are random draws from the same pool of the model's own correct traces.
      The ONLY systematic difference is whether the demonstrations obey the prompt's
      brevity instruction. A positive value is the constraint doing work.

  substitution effect random_noncompliant  -  vanilla
      Both are non-compliant demonstration sets; one is BootstrapFewShot's own pick, the
      other an arbitrary draw. A positive value means merely swapping demonstrations
      helps, with no constraint involved.

Paired exact McNemar on native-subject items for the pooled version of each contrast.
Usage: python analysis/substitution_stats.py [results/e10] [--subject ona_tili]
"""
import argparse, collections, glob, json, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact, flips

CELLS = ("random_compliant", "random_noncompliant")


def load(d):
    out = {}
    for f in sorted(glob.glob(f"{d}/*_items.jsonl")):
        items = [json.loads(l) for l in open(f)]
        if items:
            out[items[0]["model"]] = items
    return out


def acc(items, cond, subject):
    xs = [i["is_correct"] for i in items
          if i["condition"] == cond and i["subject"] == subject]
    return 100 * sum(xs) / len(xs) if xs else None


def cell_arms(items, cell):
    """Arm names belonging to a random cell, e.g. random_compliant_s0/s1/s2."""
    return sorted({i["condition"] for i in items
                   if i["condition"].startswith(cell + "_s")})


def cell_mean(items, cell, subject):
    vals = [acc(items, a, subject) for a in cell_arms(items, cell)]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paired(items, cond_a, cond_b, subject):
    """(b, c) discordant counts between two arms on one subject, paired by qid."""
    A = {i["qid"]: i for i in items
         if i["condition"] == cond_a and i["subject"] == subject}
    B = {i["qid"]: i for i in items
         if i["condition"] == cond_b and i["subject"] == subject}
    return flips([(A[q]["is_correct"], B[q]["is_correct"]) for q in A if q in B])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/e10")
    ap.add_argument("--subject", default="ona_tili")
    a = ap.parse_args()
    data = load(a.dir)
    if not data:
        sys.exit(f"no *_items.jsonl in {a.dir}")

    rows, pooled = {}, collections.Counter()
    print(f"{'model':13} {'cot':>6} {'vanilla':>8} {'compliant':>10} "
          f"{'rnd-comp':>9} {'rnd-noncomp':>12}")
    for m, items in sorted(data.items()):
        r = {"cot": acc(items, "cot", a.subject),
             "vanilla": acc(items, "dspy_bootstrap", a.subject),
             "compliant": acc(items, "dspy_bootstrap_compliant", a.subject),
             "random_compliant": cell_mean(items, "random_compliant", a.subject),
             "random_noncompliant": cell_mean(items, "random_noncompliant", a.subject)}
        r["constraint_effect"] = (
            None if r["random_compliant"] is None or r["random_noncompliant"] is None
            else round(r["random_compliant"] - r["random_noncompliant"], 1))
        r["substitution_effect"] = (
            None if r["random_noncompliant"] is None or r["vanilla"] is None
            else round(r["random_noncompliant"] - r["vanilla"], 1))
        r["arms"] = {c: cell_arms(items, c) for c in CELLS}
        rows[m] = r
        fmt = lambda v: "   --" if v is None else f"{v:6.1f}"
        print(f"{m:13} {fmt(r['cot'])} {fmt(r['vanilla']):>8} {fmt(r['compliant']):>10} "
              f"{fmt(r['random_compliant']):>9} {fmt(r['random_noncompliant']):>12}")

        # pooled paired tests, one random seed against its opposite-cell counterpart
        for ca, cb in zip(cell_arms(items, "random_compliant"),
                          cell_arms(items, "random_noncompliant")):
            b, c = paired(items, cb, ca, a.subject)   # b: noncompliant right, compliant wrong
            pooled["constraint_b"] += b
            pooled["constraint_c"] += c
        for cb in cell_arms(items, "random_noncompliant"):
            b, c = paired(items, "dspy_bootstrap", cb, a.subject)
            pooled["subst_b"] += b
            pooled["subst_c"] += c

    def summarise(key, label):
        vals = [r[key] for r in rows.values() if r[key] is not None]
        if not vals:
            return None
        mean = sum(vals) / len(vals)
        pos = sum(1 for v in vals if v > 0)
        print(f"  {label:22} mean {mean:+5.1f} pts, positive in {pos}/{len(vals)} models")
        return {"mean": round(mean, 2), "positive": pos, "n_models": len(vals),
                "per_model": vals}

    print(f"\n=== contrasts on {a.subject} (random cells averaged over seeds)")
    constraint = summarise("constraint_effect", "constraint effect")
    substitution = summarise("substitution_effect", "substitution effect")

    cb, cc = pooled["constraint_b"], pooled["constraint_c"]
    sb, sc = pooled["subst_b"], pooled["subst_c"]
    print(f"\npooled paired (exact McNemar, seed-matched draws):")
    print(f"  constraint    noncompliant->compliant   b={cb} c={cc} "
          f"p={mcnemar_exact(cb, cc):.4f}")
    print(f"  substitution  vanilla->random-noncomp   b={sb} c={sc} "
          f"p={mcnemar_exact(sb, sc):.4f}")

    verdict = "inconclusive"
    if constraint and substitution:
        if constraint["mean"] > 0 and abs(substitution["mean"]) < abs(constraint["mean"]):
            verdict = "CONSTRAINT dominates -- the mechanistic account survives"
        elif substitution["mean"] >= constraint["mean"] and substitution["mean"] > 0:
            verdict = ("SUBSTITUTION dominates -- the fix is not working through "
                       "compliance and our account of it is wrong")
    print(f"\nverdict: {verdict}")

    out = {"dir": a.dir, "subject": a.subject, "per_model": rows,
           "constraint_effect": constraint, "substitution_effect": substitution,
           "pooled": {"constraint": {"b": cb, "c": cc,
                                     "p": round(mcnemar_exact(cb, cc), 4)},
                      "substitution": {"b": sb, "c": sc,
                                       "p": round(mcnemar_exact(sb, sc), 4)}},
           "verdict": verdict}
    dest = f"{a.dir}/substitution_stats.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
