#!/usr/bin/env python3
"""Native-subject differential harm: is the native-language subject hurt MORE by
bootstrapped demonstrations than the benchmark's other subjects?

This is the contrast the original paper's claim actually implies ("native language
erodes while reasoning subjects improve"), as opposed to the absolute native-subject
McNemar, which asks only whether native accuracy fell. The two can disagree: a stack on
which bootstrapping helps everything can still leave the native subject as the sole
non-beneficiary.

Estimand: conditional on an item CHANGING its correctness between cot and vanilla
bootstrapping, the odds that the change is harmful, for native vs non-native items.
  - per model: 2x2 [[native lost, native gained], [other lost, other gained]]
  - pooled across models: Mantel-Haenszel common odds ratio (each model its own stratum,
    so between-model differences in baseline flip rate cannot drive the result)
  - uncertainty: bootstrap resampling ITEMS, not item-model cells, because all models
    answer the same items -- the naive Fisher p treats those repeats as independent and
    is anti-conservative.

`--native` also accepts a COMMA-SEPARATED GROUP of subjects, which turns the same
estimator into a subject-TYPE contrast (e.g. knowledge vs reasoning). That is the
contrast the paper's headline claim asserts: reporting "reasoning improved (p=.029)"
next to "knowledge did not (p=.150)" is two marginal tests, and the difference between
a significant and a non-significant result is not itself significant. Grouping the label
estimates the difference directly, on the same strata and the same bootstrap.

NOT PRE-REGISTERED. The pre-specified endpoint is the absolute native-subject McNemar
(analysis/decompose.py); this is a secondary analysis and must be reported as such.

Usage: python analysis/interaction.py results/e7 --native Turkish_Language_and_Literature
       python analysis/interaction.py results/e9 --native ona_tili,tarix   # subject type
"""
import argparse, glob, json, random, sys
sys.path.insert(0, ".")
from src.stats import (fisher_exact_2x2, mantel_haenszel_or,
                       heterogeneity, dersimonian_laird)

COND_A, COND_B = "cot", "dspy_bootstrap"   # defaults; override per call / via CLI


def in_group(subject, native):
    """True if `subject` carries the label under test. `native` is either a single
    subject name or a collection of them; a collection makes the contrast group-vs-rest
    (subject TYPE) instead of subject-vs-rest, with everything else unchanged."""
    return subject == native if isinstance(native, str) else subject in set(native)


def label_of(native):
    """JSON-serialisable form of the label, so a grouped run records which subjects it
    pooled rather than an unordered set repr."""
    return native if isinstance(native, str) else sorted(native)


def load_traces_dir(d):
    """Original-stack fallback: *_traces.jsonl carry no qid, so items are identified by
    POSITION within a condition -- the convention analysis/mcnemar.py already uses. Only
    valid because every condition and every model lists the same items in the same
    order; verified here rather than assumed, since a silent misalignment would pair
    unrelated questions and fabricate flips."""
    out, signature = {}, None
    for f in sorted(glob.glob(f"{d}/*_traces.jsonl")):
        model = f.split("/")[-1].replace("_traces.jsonl", "")
        by = {}
        for line in open(f):
            r = json.loads(line)
            by.setdefault(r["condition"], []).append(r)
        order = {c: [(r["subject"], r["correct"]) for r in rows] for c, rows in by.items()}
        for c, seq in order.items():
            if seq != next(iter(order.values())):
                raise SystemExit(f"{f}: condition {c} item order differs from the others")
        seq = next(iter(order.values()))
        if signature is None:
            signature = seq
        elif seq != signature:
            raise SystemExit(f"{f}: item sequence differs from the other models")
        items = []
        for cond, rows in by.items():
            for i, r in enumerate(rows):
                items.append({**r, "model": model, "qid": f"pos{i:04d}",
                              "truncated": False, "parse_error": False,
                              "rescue_correct": r["is_correct"]})
        out[model] = items
    return out


def load_dir(d):
    """{model: items}. Fails loudly on a file holding more than one model, or on a
    repeated (condition, qid): both would silently collapse into a single response and
    quietly bias the odds ratio rather than erroring."""
    out = {}
    if not glob.glob(f"{d}/*_items.jsonl"):
        return load_traces_dir(d)
    for f in sorted(glob.glob(f"{d}/*_items.jsonl")):
        items = [json.loads(l) for l in open(f)]
        if not items:
            continue
        models = {i["model"] for i in items}
        if len(models) > 1:
            raise SystemExit(f"{f}: expected one model per file, found {sorted(models)}")
        keys = [(i["condition"], i["qid"]) for i in items]
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{f}: duplicate (condition, qid) rows")
        out[items[0]["model"]] = items
    return out


def cell_contributions(data, native, cond_a=COND_A, cond_b=COND_B):
    """{qid: [(model, is_native, lost, gained), ...]} -- one entry per (model, item)
    pair that is DISCORDANT between the two conditions, keyed by item so the bootstrap
    can resample items as whole clusters."""
    rows = {}
    for model, items in data.items():
        A = {i["qid"]: i for i in items if i["condition"] == cond_a}
        B = {i["qid"]: i for i in items if i["condition"] == cond_b}
        if not A or not B:
            raise SystemExit(f"{model}: missing condition {cond_a!r} or {cond_b!r} "
                             f"(have {sorted({i['condition'] for i in items})})")
        for q, a in A.items():
            b = B.get(q)
            if b is None:
                continue
            lost = int(bool(a["is_correct"]) and not b["is_correct"])
            gained = int(not a["is_correct"] and bool(b["is_correct"]))
            rows.setdefault(q, []).append(
                (model, in_group(a["subject"], native), lost, gained))
    return rows


def strata_from(rows, qids):
    """[(native_lost, native_gained, other_lost, other_gained)] per model."""
    per_model = {}
    for q in qids:
        for model, is_native, lost, gained in rows[q]:
            s = per_model.setdefault(model, [0, 0, 0, 0])
            if is_native:
                s[0] += lost
                s[1] += gained
            else:
                s[2] += lost
                s[3] += gained
    return per_model


def analyse(d, native, n_boot=4000, seed=42, cond_a=COND_A, cond_b=COND_B,
            exclude_models=()):
    data = load_dir(d)
    if not data:
        return None
    # Drop models the caller knows are untreated. Zero-discordance detection is not
    # enough: a model whose program the optimizer left unchanged can still produce
    # discordant pairs through its own run-to-run nondeterminism, and every one of
    # those flips is noise rather than treatment effect.
    excluded = [m for m in data if m in set(exclude_models)]
    for m in excluded:
        del data[m]
    if not data:
        return None
    rows = cell_contributions(data, native, cond_a, cond_b)
    qids = sorted(rows)
    per_model = strata_from(rows, qids)

    out = {"dir": d, "native": label_of(native), "contrast": f"{cond_a} -> {cond_b}",
           "n_models": len(data), "n_items": len(qids), "excluded_models": excluded,
           "per_model": {}, "preregistered": False}
    for m, (bn, cn, bo, co) in sorted(per_model.items()):
        out["per_model"][m] = {
            "native_lost": bn, "native_gained": cn,
            "other_lost": bo, "other_gained": co,
            # A model with zero discordant pairs contributes nothing to the pooled odds
            # ratio. That is a STRUCTURAL null (the optimizer changed nothing, or changed
            # nothing that flipped an item), not evidence of no harm, and must not be
            # read as a measured null.
            "discordant_pairs": bn + cn + bo + co,
            "native_harm_share": round(100 * bn / (bn + cn), 1) if bn + cn else None,
            "other_harm_share": round(100 * bo / (bo + co), 1) if bo + co else None,
            "odds_ratio": round(bn * co / (cn * bo), 2) if cn and bo else None,
        }
    out["n_informative"] = sum(1 for r in out["per_model"].values()
                               if r["discordant_pairs"] > 0)
    out["uninformative_models"] = sorted(m for m, r in out["per_model"].items()
                                         if r["discordant_pairs"] == 0)

    tot = [sum(v[i] for v in per_model.values()) for i in range(4)]
    out["pooled"] = {"native_lost": tot[0], "native_gained": tot[1],
                     "other_lost": tot[2], "other_gained": tot[3],
                     "fisher_p_unclustered": round(fisher_exact_2x2(*tot), 5)}
    point = mantel_haenszel_or(list(per_model.values()))
    out["mh_odds_ratio"] = round(point, 3)

    # item-cluster bootstrap: resample items with replacement, carrying every model's
    # response for a drawn item along with it
    rng = random.Random(seed)
    boots, re_boots = [], []
    for _ in range(n_boot):
        samp = [rng.choice(qids) for _ in qids]
        st = list(strata_from(rows, samp).values())
        v = mantel_haenszel_or(st)
        if v != float("inf"):
            boots.append(v)
        # The DL interval assumes strata are INDEPENDENT. They are not -- every model
        # answers the same items -- so the analytic interval is anti-conservative in
        # exactly the way the naive Fisher p is. Re-estimating DL inside the item
        # cluster bootstrap gives an interval that respects both the between-model
        # heterogeneity and the shared items.
        r = dersimonian_laird(st)["or"]
        if r == r and r not in (float("inf"), 0.0):
            re_boots.append(r)
    boots.sort()
    re_boots.sort()
    # always present, even when no draw yielded a finite OR (no non-native discordance)
    out["bootstrap"] = {"n": len(boots), "seed": seed, "ci95": None, "p_or_le_1": None}
    if boots:
        out["bootstrap"]["ci95"] = [round(boots[int(0.025 * len(boots))], 3),
                                    round(boots[int(0.975 * len(boots))], 3)]
        out["bootstrap"]["p_or_le_1"] = round(
            sum(1 for x in boots if x <= 1.0) / len(boots), 4)
    # Does the common-odds-ratio assumption behind the MH estimate actually hold?
    strata_list = list(per_model.values())
    out["heterogeneity"] = {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in heterogeneity(strata_list).items()}
    dl = dersimonian_laird(strata_list)
    out["random_effects"] = {
        "or": round(dl["or"], 3),
        "ci95_analytic": [round(dl["ci95"][0], 3), round(dl["ci95"][1], 3)],
        "tau2": round(dl["tau2"], 4),
        "ci95_cluster_boot": ([round(re_boots[int(0.025 * len(re_boots))], 3),
                               round(re_boots[int(0.975 * len(re_boots))], 3)]
                              if re_boots else None),
        "p_or_le_1_cluster_boot": (round(sum(1 for x in re_boots if x <= 1.0)
                                         / len(re_boots), 4) if re_boots else None),
    }

    # how many models point the same way (direction consistency, not a formal test)
    dirs = [r for r in out["per_model"].values()
            if r["native_harm_share"] is not None and r["other_harm_share"] is not None]
    out["models_agreeing"] = sum(1 for r in dirs
                                 if r["native_harm_share"] > r["other_harm_share"])
    out["models_comparable"] = len(dirs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default="results/e7")
    ap.add_argument("--native", default="Turkish_Language_and_Literature")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--cond-a", default=COND_A)
    ap.add_argument("--cond-b", default=COND_B, help="e.g. dspy_mipro for E8")
    ap.add_argument("--exclude", default="",
                    help="comma list of models to drop (e.g. ones the optimizer no-oped)")
    ap.add_argument("--out", default=None,
                    help="destination JSON (default <dir>/interaction.json); set it for "
                         "grouped runs so they do not overwrite the single-subject one")
    a = ap.parse_args()
    ex = [m.strip() for m in a.exclude.split(",") if m.strip()]
    # a comma in --native means a subject GROUP: the contrast becomes group-vs-rest
    parts = [x.strip() for x in a.native.split(",") if x.strip()]
    native = parts[0] if len(parts) == 1 else parts
    out = analyse(a.dir, native, n_boot=a.boot, cond_a=a.cond_a, cond_b=a.cond_b,
                  exclude_models=ex)
    if out is None:
        sys.exit(f"no *_items.jsonl in {a.dir}")

    print(f"=== differential harm, label={'+'.join(parts)}, {out['contrast']} "
          f"({out['n_models']} models, {out['n_items']} items) [NOT pre-registered]")
    print(f"{'model':14} {'lbl L/G':>10} {'lbl harm%':>10} "
          f"{'oth L/G':>10} {'oth harm%':>10} {'OR':>7}")
    for m, r in out["per_model"].items():
        nat = "{}/{}".format(r["native_lost"], r["native_gained"])
        oth = "{}/{}".format(r["other_lost"], r["other_gained"])
        print(f"{m:14} {nat:>10} {str(r['native_harm_share']):>10} "
              f"{oth:>10} {str(r['other_harm_share']):>10} {str(r['odds_ratio']):>7}")
    p = out["pooled"]
    print(f"\npooled  label {p['native_lost']}/{p['native_gained']}  "
          f"other {p['other_lost']}/{p['other_gained']}  "
          f"Fisher p={p['fisher_p_unclustered']} (unclustered, anti-conservative)")
    print(f"Mantel-Haenszel OR (model-stratified) = {out['mh_odds_ratio']}")
    b = out["bootstrap"]
    if b["ci95"] is None:
        print(f"item-cluster bootstrap: no finite draws ({b['n']}) -- OR unidentified")
    else:
        print(f"item-cluster bootstrap ({b['n']} draws, seed {b['seed']}): "
              f"95% CI {b['ci95']}  P(OR<=1)={b['p_or_le_1']}")
    print(f"direction consistent in {out['models_agreeing']}/{out['models_comparable']} models")
    h, re = out["heterogeneity"], out["random_effects"]
    if h["Q"] is not None:
        print(f"\nheterogeneity: Q={h['Q']:.2f} (df={h['df']}, p={h['p']:.4f})  "
              f"I2={h['I2']:.1f}%  tau2={h['tau2']:.4f}")
        print(f"random effects (DerSimonian-Laird): OR {re['or']}  "
              f"analytic CI {re['ci95_analytic']}")
        print(f"   + item-cluster bootstrap CI {re['ci95_cluster_boot']}  "
              f"P(OR<=1)={re['p_or_le_1_cluster_boot']}")
    if out["excluded_models"]:
        print(f"EXCLUDED as untreated: {', '.join(out['excluded_models'])}")
    if out["uninformative_models"]:
        print(f"NOTE: {len(out['uninformative_models'])} of {out['n_models']} models "
              f"contributed ZERO discordant pairs and are structurally uninformative "
              f"(not measured nulls): {', '.join(out['uninformative_models'])}")

    dest = a.out or f"{a.dir}/interaction.json"
    json.dump(out, open(dest, "w"), indent=1)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
