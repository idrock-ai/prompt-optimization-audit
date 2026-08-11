#!/usr/bin/env python3
"""E1 decomposition: classify every cot-correct -> bootstrap-wrong flip on ona_tili as
truncation / format_drift / content; dual-scored pooled McNemar; ladder table.
Usage: python analysis/decompose.py [results/e1] [--subject ona_tili]"""
import argparse, collections, glob, json, sys
sys.path.insert(0, ".")
from src.stats import mcnemar_exact, flips


def classify_flip(cot_item, boot_item):
    if boot_item["truncated"]:
        return "truncation"
    if boot_item["parse_error"] or not boot_item["predicted"]:
        return "format_drift"
    return "content"


def _pairs(items, cond_a, cond_b, subject):
    A = {i["qid"]: i for i in items if i["condition"] == cond_a and i["subject"] == subject}
    B = {i["qid"]: i for i in items if i["condition"] == cond_b and i["subject"] == subject}
    return [(A[q], B[q]) for q in A if q in B]


def decompose_dir(d, subject="ona_tili", cond_a="cot", cond_b="dspy_bootstrap"):
    out = {"per_model": {}, "pooled": {}}
    pooled_pairs, clean_pairs, rescue_pairs = [], [], []
    for f in sorted(glob.glob(f"{d}/*_items.jsonl")):
        items = [json.loads(l) for l in open(f)]
        model = items[0]["model"]
        pairs = _pairs(items, cond_a, cond_b, subject)
        cls = collections.Counter(classify_flip(a, b) for a, b in pairs
                                  if a["is_correct"] and not b["is_correct"])
        b_, c_ = flips([(a["is_correct"], b["is_correct"]) for a, b in pairs])
        out["per_model"][model] = {"flips": {k: cls.get(k, 0) for k in
                                             ("truncation", "format_drift", "content")},
                                   "b": b_, "c": c_, "n": len(pairs)}
        pooled_pairs += [(a["is_correct"], b["is_correct"]) for a, b in pairs]
        clean_pairs += [(a["is_correct"], b["is_correct"]) for a, b in pairs
                        if not (a["parse_error"] or b["parse_error"]
                                or a["truncated"] or b["truncated"])]
        rescue_pairs += [(a["rescue_correct"], b["rescue_correct"]) for a, b in pairs]
    for name, prs in (("deployment", pooled_pairs), ("knowledge", clean_pairs),
                      ("rescue", rescue_pairs)):
        b_, c_ = flips(prs)
        out["pooled"][name + "_p"] = round(mcnemar_exact(b_, c_), 4)
        out["pooled"][name] = {"b": b_, "c": c_, "n": len(prs)}
    out["pooled"]["b"], out["pooled"]["c"] = flips(pooled_pairs)

    # How much of the harm the format route actually accounts for. Reported explicitly
    # because it is the number that bounds the mechanism's claim: a mechanism that
    # explains a minority of the harmful flips is a real mechanism, but it is not the
    # whole story, and a reader should not have to derive that share from a figure.
    tot = collections.Counter()
    for r in out["per_model"].values():
        tot.update(r["flips"])
    harmful = sum(tot.values())
    fmt = tot["truncation"] + tot["format_drift"]
    out["pooled"]["flip_classes"] = dict(tot)
    out["pooled"]["harmful_flips"] = harmful
    out["pooled"]["format_flips"] = fmt
    out["pooled"]["format_share_pct"] = round(100 * fmt / harmful, 1) if harmful else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/e1")
    ap.add_argument("--subject", default="ona_tili")
    a = ap.parse_args()
    out = decompose_dir(a.dir, a.subject)
    print(f"{'model':14} {'trunc':>6} {'format':>7} {'content':>8} {'b':>4} {'c':>4}")
    for m, r in out["per_model"].items():
        f = r["flips"]
        print(f"{m:14} {f['truncation']:>6} {f['format_drift']:>7} "
              f"{f['content']:>8} {r['b']:>4} {r['c']:>4}")
    for k in ("deployment", "knowledge", "rescue"):
        s = out["pooled"][k]
        print(f"POOLED {k:11} b={s['b']:>3} c={s['c']:>3} n={s['n']:>5} "
              f"exact-McNemar p={out['pooled'][k + '_p']}")
    json.dump(out, open(f"{a.dir}/decomposition.json", "w"), indent=1)


if __name__ == "__main__":
    main()
