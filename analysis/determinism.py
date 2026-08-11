#!/usr/bin/env python3
"""Run-to-run determinism at temperature 0, measured from two independent sessions.

E1 and E8 both evaluate the SAME CoT program on the SAME 251 items, same machine, same
serving stack, greedy decoding. Any disagreement between them is pure session noise, and
it sets a floor beneath every per-model cell the paper reports.

Two quantities, because they differ by a lot and conflating them either overstates or
understates the problem:

  item churn    how many items change correctness between sessions. This is what bounds
                claims about individual items or about small flip counts.
  net drift     how much the model's ACCURACY moves. Much smaller than churn, because
                flips in the two directions largely cancel. This is what bounds claims
                about the deltas in the paper's tables.

Usage: python analysis/determinism.py [--a results/e1] [--b results/e8]
"""
import argparse, collections, glob, json, sys
sys.path.insert(0, ".")


def load_cot(d):
    """{(model, qid): item} for the CoT condition."""
    out = {}
    for f in glob.glob(f"{d}/*_items.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if r["condition"] == "cot":
                out[(r["model"], r["qid"])] = r
    return out


def compare(a, b, subject=None):
    """{model: {churn, n, acc_a, acc_b}} over items shared by both sessions."""
    agg = collections.defaultdict(lambda: {"churn": 0, "n": 0, "k_a": 0, "k_b": 0})
    for key, ra in a.items():
        rb = b.get(key)
        if rb is None or (subject and ra["subject"] != subject):
            continue
        s = agg[key[0]]
        s["n"] += 1
        s["k_a"] += bool(ra["is_correct"])
        s["k_b"] += bool(rb["is_correct"])
        s["churn"] += int(bool(ra["is_correct"]) != bool(rb["is_correct"]))
    out = {}
    for m, s in agg.items():
        n = max(s["n"], 1)
        out[m] = {"n": s["n"], "churn": s["churn"],
                  "churn_pct": round(100 * s["churn"] / n, 1),
                  "acc_a": round(100 * s["k_a"] / n, 1),
                  "acc_b": round(100 * s["k_b"] / n, 1),
                  "net_drift": round(100 * (s["k_b"] - s["k_a"]) / n, 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results/e1")
    ap.add_argument("--b", default="results/e8")
    ap.add_argument("--subject", default="ona_tili")
    ap.add_argument("--out", default="results/determinism.json")
    args = ap.parse_args()

    A, B = load_cot(args.a), load_cot(args.b)
    if not A or not B:
        sys.exit(f"no CoT items in {args.a} or {args.b}")
    overall = compare(A, B)
    native = compare(A, B, subject=args.subject)

    print(f"CoT re-run, {args.a} vs {args.b} (same stack, temperature 0, same program)")
    print(f"{'model':13} {'all: churn':>13} {'drift':>7}   "
          f"{args.subject + ': churn':>20} {'drift':>7}")
    for m in sorted(overall):
        o, s = overall[m], native.get(m, {})
        print(f"{m:13} {o['churn']:>5}/{o['n']:<7} {o['net_drift']:>+6.1f}   "
              f"{s.get('churn', 0):>13}/{s.get('n', 0):<6} {s.get('net_drift', 0):>+6.1f}")

    det = sorted(m for m, o in overall.items() if o["churn"] == 0)
    nondet = sorted(m for m, o in overall.items() if o["churn"] > 0)
    print(f"\ndeterministic: {', '.join(det) or 'none'}")
    print(f"NOT deterministic: {', '.join(nondet) or 'none'}")
    if nondet:
        worst_churn = max(native.get(m, {}).get("churn_pct", 0) for m in nondet)
        worst_drift = max(abs(native.get(m, {}).get("net_drift", 0)) for m in nondet)
        print(f"worst {args.subject} item churn {worst_churn}%; "
              f"worst {args.subject} net accuracy drift {worst_drift} points "
              f"-- the floor beneath any per-model delta we report for these models.")

    json.dump({"a": args.a, "b": args.b, "subject": args.subject,
               "overall": overall, "native": native,
               "deterministic": det, "nondeterministic": nondet},
              open(args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
