#!/usr/bin/env python3
"""E2 dose-response: erosion and truncation vs max_tokens.
Reads results/e2/mt{256,1024,2048}/ plus results/e1 (the 512 cell) unless a single
root with mtNNN/ subdirs is given.

Three tests, deliberately separated, because they do not carry the same weight:

  1. MANIPULATION CHECK (Cochran-Armitage on truncation counts). Raising max_tokens
     mechanically reduces the number of generations that hit the cap, so a significant
     trend here confirms the knob turned -- it is not evidence about accuracy. Two
     further caveats travel with it: the four budget cells score the SAME items, so the
     independent-groups assumption behind Cochran-Armitage is violated and the nominal
     p is anti-conservative; and the trend need not be monotone (qwen3.5:9b is not).

  2. OUTCOME TEST (paired exact McNemar, bootstrap@256 vs bootstrap@2048, same items).
     This is the causal question that matters: does relieving the budget improve native
     accuracy under vanilla bootstrapping? It is properly paired -- same model, same
     items, one knob changed.

  3. RESIDUAL. The erosion still present at 2048 with zero truncations, which no amount
     of budget can be responsible for.

Usage: python analysis/dose_response.py [root]
"""
import argparse, glob, json, sys
sys.path.insert(0, ".")
from src.stats import cochran_armitage, mcnemar_exact


def _load(d):
    items = []
    for f in glob.glob(f"{d}/*_items.jsonl"):
        items += [json.loads(l) for l in open(f)]
    return items


def collect(root, budgets=(256, 512, 1024, 2048), e1_dir=None):
    table, paired = {}, {}
    for mt in budgets:
        d = f"{root}/mt{mt}"
        items = _load(d)
        if not items and mt == 512 and e1_dir:      # 512 cell lives in results/e1
            items = [i for i in _load(e1_dir)]
        for i in items:
            if i["subject"] != "ona_tili":
                continue
            m = table.setdefault(i["model"], {})
            cell = m.setdefault(i["max_tokens"], {"cot_k": 0, "cot_n": 0, "boot_k": 0,
                                                  "boot_n": 0, "trunc_boot": 0})
            if i["condition"] == "cot":
                cell["cot_k"] += i["is_correct"]; cell["cot_n"] += 1
            elif i["condition"] == "dspy_bootstrap":
                cell["boot_k"] += i["is_correct"]; cell["boot_n"] += 1
                cell["trunc_boot"] += int(i["truncated"])
                # keep the per-item outcome so the extremes can be paired by qid
                paired.setdefault(i["model"], {}).setdefault(
                    i["max_tokens"], {})[i["qid"]] = bool(i["is_correct"])

    for m, row in table.items():
        bs = [b for b in budgets if b in row]
        ks = [row[b]["trunc_boot"] for b in bs]
        ns = [max(row[b]["boot_n"], 1) for b in bs]
        _, p = cochran_armitage(ks, ns)
        row["trend_p"] = p                                   # manipulation check
        row["monotone"] = all(x >= y for x, y in zip(ks, ks[1:]))

        # outcome test: bootstrap@lowest vs bootstrap@highest budget, paired by item.
        # Only defined for models actually swept across budgets; the models present at a
        # single budget would otherwise be compared against themselves.
        if len(bs) > 1:
            lo, hi = paired.get(m, {}).get(bs[0], {}), paired.get(m, {}).get(bs[-1], {})
            shared = sorted(set(lo) & set(hi))
            b = sum(1 for q in shared if lo[q] and not hi[q])
            c = sum(1 for q in shared if not lo[q] and hi[q])
            row["outcome"] = {"lo_budget": bs[0], "hi_budget": bs[-1],
                              "n_paired": len(shared),
                              "b_lo_right_hi_wrong": b, "c_lo_wrong_hi_right": c,
                              "p": round(mcnemar_exact(b, c), 4)}
        else:
            row["outcome"] = None
        # residual: delta and truncations at the highest budget
        hic = row[bs[-1]]
        row["residual"] = {
            "budget": bs[-1],
            "delta": round(100 * hic["boot_k"] / max(hic["boot_n"], 1)
                           - 100 * hic["cot_k"] / max(hic["cot_n"], 1), 1),
            "truncations": hic["trunc_boot"]}
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="results/e2")
    ap.add_argument("--e1", default="results/e1")
    a = ap.parse_args()
    table = collect(a.root, e1_dir=a.e1)
    for m, row in table.items():
        mono = "monotone" if row["monotone"] else "NOT monotone"
        print(f"== {m}")
        for b in sorted(k for k in row if isinstance(k, int)):
            c = row[b]
            ca = 100 * c["cot_k"] / max(c["cot_n"], 1)
            ba = 100 * c["boot_k"] / max(c["boot_n"], 1)
            print(f"  mt={b:<5} cot={ca:5.1f} boot={ba:5.1f} d={ba - ca:+5.1f} "
                  f"trunc(boot)={c['trunc_boot']}")
        print(f"  [1] truncation trend (manipulation check, shared items -> "
              f"anti-conservative): p={row['trend_p']:.2g}, {mono}")
        o = row["outcome"]
        if o:
            print(f"  [2] OUTCOME boot@{o['lo_budget']} vs boot@{o['hi_budget']}, "
                  f"paired: b={o['b_lo_right_hi_wrong']} c={o['c_lo_wrong_hi_right']} "
                  f"n={o['n_paired']} p={o['p']}")
        else:
            print("  [2] OUTCOME: not swept across budgets (single cell)")
        print(f"  [3] residual at {row['residual']['budget']}: "
              f"delta={row['residual']['delta']:+.1f} with "
              f"{row['residual']['truncations']} truncations")

    # Pooled outcome test over the models that actually truncate. Pre-specifiable
    # without looking at accuracy: a budget can only help a model whose generations hit
    # the cap, so the treated set is defined by truncation counts, not by outcome.
    treated = [m for m, r in table.items()
               if r["outcome"] and max(r[b]["trunc_boot"] for b in r if isinstance(b, int)) > 0]
    ctrl = [m for m, r in table.items()
            if r["outcome"] and m not in treated]
    for label, group in (("treated (truncating models)", treated), ("control", ctrl)):
        if not group:
            continue
        b = sum(table[m]["outcome"]["b_lo_right_hi_wrong"] for m in group)
        c = sum(table[m]["outcome"]["c_lo_wrong_hi_right"] for m in group)
        n = sum(table[m]["outcome"]["n_paired"] for m in group)
        print(f"\nPOOLED OUTCOME, {label} ({', '.join(sorted(group))}): "
              f"b={b} c={c} n={n} exact McNemar p={mcnemar_exact(b, c):.4f}")
        table.setdefault("_pooled_outcome", {})[label] = {
            "models": sorted(group), "b": b, "c": c, "n": n,
            "p": round(mcnemar_exact(b, c), 4)}
    json.dump(table, open(f"{a.root}/dose_response.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
