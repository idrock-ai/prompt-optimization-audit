#!/usr/bin/env python3
"""E5 fixes table: ona_tili recovery vs math retention for each mitigation.
Sources: results/e1 (vanilla/compliant/rescue @512), results/e2/mt2048 + results/e5/mt2048.

TWO ESTIMATORS, and the paper reports both, because they disagree.

  per-model mean  the pre-registered one: compute recovery/retention per model, then
                  average. Its weaknesses are structural, not incidental: a model with
                  no erosion has nothing to recover, so `recovery` awards it a
                  DEFINITIONAL 100 regardless of what the fix did; a model that
                  overshoots CoT scores above 100 and drags the mean up with it; and
                  models whose math gain is a single item carry denominators near zero.

  pooled          points recovered / points lost, summed over the models that actually
                  eroded (and likewise for math gain). One erosion point counts the same
                  wherever it occurred, definitional 100s cannot enter, and no
                  small denominator can dominate.

The pooled figure is the one to believe. We keep the per-model mean because it is what
we pre-registered the >=80/>=90 bar against, and reporting only the estimator that
passes would be exactly the practice this paper criticises.

Usage: python analysis/fixes_table.py
"""
import glob, json, sys
sys.path.insert(0, ".")


def _acc(items, cond, subject, field="is_correct"):
    xs = [i[field] for i in items if i["condition"] == cond and i["subject"] == subject]
    return 100 * sum(xs) / len(xs) if xs else None


def recovery(cot, vanilla, fixed):
    """% of the cot->vanilla erosion recovered by the fix.

    Returns None -- not 100 -- when there was no erosion to recover. The caller decides
    whether to score that as a definitional 100 (the pre-registered per-model mean) or
    to leave it out of the numerator and denominator alike (the pooled estimator)."""
    lost = cot - vanilla
    if lost <= 0:
        return None
    return 100 * (fixed - vanilla) / lost


def retention(cot, vanilla, fixed):
    """% of the cot->vanilla math gain retained by the fix. None when there was no gain."""
    gain = vanilla - cot
    if gain <= 0:
        return None
    return 100 * (fixed - cot) / gain


def load_dir(d):
    out = {}
    for f in glob.glob(f"{d}/*_items.jsonl"):
        items = [json.loads(l) for l in open(f)]
        out[items[0]["model"]] = items
    return out


def build_rows():
    e1 = load_dir("results/e1")
    mt2048 = {**load_dir("results/e2/mt2048"), **load_dir("results/e5/mt2048")}
    rows = []
    for m, items in sorted(e1.items()):
        cot_o = _acc(items, "cot", "ona_tili")
        cot_m = _acc(items, "cot", "matematika")
        van_o = _acc(items, "dspy_bootstrap", "ona_tili")
        van_m = _acc(items, "dspy_bootstrap", "matematika")
        fixes = {
            "compliant": (_acc(items, "dspy_bootstrap_compliant", "ona_tili"),
                          _acc(items, "dspy_bootstrap_compliant", "matematika")),
            "rescue": (_acc(items, "dspy_bootstrap", "ona_tili", "rescue_correct"),
                       _acc(items, "dspy_bootstrap", "matematika", "rescue_correct")),
            "budget2048": (_acc(mt2048.get(m, []), "dspy_bootstrap", "ona_tili"),
                           _acc(mt2048.get(m, []), "dspy_bootstrap", "matematika")),
            "compliant+rescue": (
                _acc(items, "dspy_bootstrap_compliant", "ona_tili", "rescue_correct"),
                _acc(items, "dspy_bootstrap_compliant", "matematika", "rescue_correct")),
        }
        row = {"model": m, "cot_ona": cot_o, "vanilla_ona": van_o,
               "cot_math": cot_m, "vanilla_math": van_m,
               "eroded": cot_o - van_o > 0, "math_gained": van_m - cot_m > 0,
               "fixes": {}}
        for name, (fo, fm) in fixes.items():
            if fo is None:
                continue
            row["fixes"][name] = {
                "ona": fo, "math": fm,
                "recovery": recovery(cot_o, van_o, fo),
                "retention": retention(cot_m, van_m, fm),
                # points, for the pooled estimator
                "ona_points_lost": max(0.0, cot_o - van_o),
                "ona_points_back": (fo - van_o) if cot_o - van_o > 0 else 0.0,
                "math_points_gained": max(0.0, van_m - cot_m),
                "math_points_kept": (fm - cot_m) if van_m - cot_m > 0 else 0.0,
            }
        rows.append(row)
    return rows


def summarise(rows):
    """{fix: {per_model_mean_*, pooled_*, n_definitional_*}}."""
    out = {}
    names = [n for r in rows for n in r["fixes"]]
    for name in dict.fromkeys(names):
        fs = [r["fixes"][name] for r in rows if name in r["fixes"]]
        # pre-registered per-model mean: None (no erosion / no gain) scores 100
        rec_mean = sum(100.0 if f["recovery"] is None else f["recovery"] for f in fs) / len(fs)
        ret_mean = sum(100.0 if f["retention"] is None else f["retention"] for f in fs) / len(fs)
        lost = sum(f["ona_points_lost"] for f in fs)
        back = sum(f["ona_points_back"] for f in fs)
        gained = sum(f["math_points_gained"] for f in fs)
        kept = sum(f["math_points_kept"] for f in fs)
        out[name] = {
            "per_model_mean_recovery": round(rec_mean, 1),
            "per_model_mean_retention": round(ret_mean, 1),
            "pooled_recovery": round(100 * back / lost, 1) if lost else None,
            "pooled_retention": round(100 * kept / gained, 1) if gained else None,
            "ona_points_lost": round(lost, 1), "ona_points_back": round(back, 1),
            "math_points_gained": round(gained, 1), "math_points_kept": round(kept, 1),
            "n_models": len(fs),
            "n_definitional_recovery_100": sum(1 for f in fs if f["recovery"] is None),
            "n_recovery_over_100": sum(1 for f in fs
                                       if f["recovery"] is not None and f["recovery"] > 100),
            "n_definitional_retention_100": sum(1 for f in fs if f["retention"] is None),
            "bar_per_model_mean": "PASS" if rec_mean >= 80 and ret_mean >= 90 else "miss",
            "bar_pooled": ("PASS" if lost and gained and 100 * back / lost >= 80
                           and 100 * kept / gained >= 90 else "miss"),
        }
    return out


def main():
    rows = build_rows()
    fmt = lambda v: "  --" if v is None else f"{v:4.0f}"
    print(f"{'model':13} {'fix':17} {'ona':>6} {'recov%':>7} {'math':>6} {'reten%':>7}")
    for r in rows:
        print(f"{r['model']:13} {'vanilla':17} {r['vanilla_ona']:>6.1f} {'':>7} "
              f"{r['vanilla_math']:>6.1f}")
        for name, f in r["fixes"].items():
            print(f"{'':13} {name:17} {f['ona']:>6.1f} {fmt(f['recovery']):>7} "
                  f"{f['math']:>6.1f} {fmt(f['retention']):>7}")
    print("   ('--' = nothing to recover / no gain to retain; the per-model mean scores "
          "these as 100)")

    summary = summarise(rows)
    print("\n" + "=" * 78)
    print("PRE-REGISTERED per-model mean (bar: recovery >= 80 and retention >= 90)")
    for name, s in summary.items():
        print(f"  {name:17} recovery={s['per_model_mean_recovery']:5.1f} "
              f"retention={s['per_model_mean_retention']:5.1f}  {s['bar_per_model_mean']}"
              f"   [{s['n_definitional_recovery_100']} definitional 100s, "
              f"{s['n_recovery_over_100']} over-100 overshoots]")
    print("\nPOOLED, points-based (same bar) -- the estimator to believe")
    for name, s in summary.items():
        print(f"  {name:17} recovery={s['pooled_recovery']:5.1f} "
              f"({s['ona_points_back']}/{s['ona_points_lost']} pts)   "
              f"retention={s['pooled_retention']:5.1f} "
              f"({s['math_points_kept']}/{s['math_points_gained']} pts)  {s['bar_pooled']}")

    json.dump({"rows": rows, "summary": summary},
              open("results/e5/fixes.json", "w"), indent=1)
    print("\nwrote results/e5/fixes.json")


if __name__ == "__main__":
    main()
