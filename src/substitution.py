"""E10: demonstration-substitution ablation -- does the fix work because the
demonstrations COMPLY, or merely because they are DIFFERENT?

The open tension the paper currently concedes: the format route accounts for ~11% of
harmful native flips, yet the compliant metric recovers ~76% of the erosion. So the fix
cannot be working through truncation removal alone. Requiring compliance changes two
things at once -- the demonstrations now satisfy the brevity instruction (CONSTRAINT),
and they are a different set of training examples (SUBSTITUTION). Nothing in the paper
separates those.

This is the 2x2 that does. From the model's own correct CoT traces over the train pool
we draw four demonstrations at random, K times, from each side of the compliance split:

  random_compliant_sN      random draw, brevity-compliant   (constraint, not BSFS's picks)
  random_noncompliant_sN   random draw, non-compliant       (substitution, no constraint)

against the two arms the paper already has:

  dspy_bootstrap             BSFS's own correctness-selected demos  (vanilla)
  dspy_bootstrap_compliant   BSFS's own correctness+brevity demos   (the fix)

Reading the result:
  random_compliant ~ compliant AND random_noncompliant ~ vanilla
      -> COMPLIANCE does the work; the identity of the demonstrations is incidental,
         and the paper's mechanistic story survives.
  both random arms ~ each other and both beat vanilla
      -> SUBSTITUTION does the work; any reshuffle of demonstrations helps and the
         constraint is not the active ingredient. That would refute our account of our
         own fix, and we would have to say so.

Several seeds per cell, because a single four-demo draw is itself noisy and we would
otherwise be comparing one arbitrary draw against another.

Usage: python -m src.substitution --model qwen3.5:9b --out-dir results/e10
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import dspy

from .data import load_splits, cap_per_subject, replication_all
from .program import CoTSolver, is_compliant, demo_field, demo_payload
from .instrument import instrumented_eval
from .run import make_lm_factory, build_items, score_items

REPO = Path(__file__).resolve().parent.parent


def make_demo(ex, reasoning):
    d = dspy.Example(question=ex.question, options=ex.options, reasoning=reasoning,
                     answer_letter=str(ex.answer_letter).strip().upper()
                     ).with_inputs("question", "options")
    d.subject = ex.subject
    return d


def collect_correct(pool, factory, workers, max_tokens):
    """The model's own correct, well-formed CoT traces over the train pool."""
    recs = instrumented_eval(pool, CoTSolver(), factory, workers, "collect-demos",
                             max_tokens=max_tokens)
    out = []
    for ex, r in zip(pool, recs):
        ok = r["predicted"] == str(ex.answer_letter).strip().upper()
        if ok and r["reasoning"] and not r["error_type"]:
            out.append(make_demo(ex, r["reasoning"]))
    return out


def split_by_compliance(correct):
    compliant = [d for d in correct if is_compliant(d.reasoning)]
    noncompliant = [d for d in correct if not is_compliant(d.reasoning)]
    return compliant, noncompliant


def draw(cands, k, seed):
    """k demos drawn without replacement, deterministically under `seed`."""
    c = sorted(cands, key=lambda d: (d.subject, d.question))
    random.Random(seed).shuffle(c)
    return c[:k]


def load_saved(path):
    if not path.is_file():
        return None
    prog = CoTSolver()
    prog.load(str(path))
    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=40)
    ap.add_argument("--k-demos", type=int, default=4)
    ap.add_argument("--n-seeds", type=int, default=3,
                    help="random draws per compliance cell")
    ap.add_argument("--e1-dir", default="results/e1")
    # --- E11 knobs. Defaults reproduce E10 exactly. ---
    ap.add_argument("--eval-set", choices=("dtm", "replication"), default="dtm",
                    help="dtm = the 251-item benchmark test split (E10). replication = "
                         "the frozen four-subject public corpus (E11), which powers the "
                         "contrast the way E9 powered the differential.")
    ap.add_argument("--cap-nonnative", type=int, default=None,
                    help="replication only: cap each NON-native subject (native is "
                         "never thinned)")
    ap.add_argument("--random-pool", choices=("split", "all"), default="split",
                    help="split = draw separately from the compliant and non-compliant "
                         "pools (E10, to isolate the constraint). all = draw from the "
                         "whole correct pool (E11: compliance is already settled as "
                         "null, so the live question is only BSFS-picked vs random).")
    ap.add_argument("--arms", default="cot,vanilla,compliant",
                    help="which reference arms to run alongside the random draws")
    ap.add_argument("--out-dir", default="results/e10")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())
    train, _dev, dtm_test = load_splits(seed=args.seed)
    test = (dtm_test if args.eval_set == "dtm"
            else replication_all(cap_nonnative=args.cap_nonnative))
    pool = cap_per_subject(train, args.train_cap)
    random.Random(args.seed).shuffle(pool)

    correct = collect_correct(pool, factory, args.workers, args.max_tokens)
    compliant, noncompliant = split_by_compliance(correct)
    print(f"[{args.model}] correct traces={len(correct)} "
          f"compliant={len(compliant)} noncompliant={len(noncompliant)}")
    for name, cands in (("compliant", compliant), ("noncompliant", noncompliant)):
        if len(cands) < args.k_demos:
            print(f"  WARNING: only {len(cands)} {name} traces available for "
                  f"{args.k_demos}-demo draws -- that cell is under-supplied and its "
                  f"result must be reported with the caveat, as E3's was.")

    safe = args.model.replace("/", "_").replace(":", "_")
    e1 = REPO / args.e1_dir
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    want = {a.strip() for a in args.arms.split(",") if a.strip()}
    arms = [("cot", [])] if "cot" in want else []
    for key, cond, fname in (("vanilla", "dspy_bootstrap", f"{safe}_bootstrap.json"),
                             ("compliant", "dspy_bootstrap_compliant",
                              f"{safe}_bootstrap_compliant.json")):
        if key not in want:
            continue
        prog = load_saved(e1 / fname)
        if prog is None:
            print(f"  NOTE: {fname} absent in {e1}; skipping the {cond} reference arm")
            continue
        arms.append((cond, list(getattr(prog.predict, "demos", []))))
    for s in range(args.n_seeds):
        seed = args.seed + 1000 * (s + 1)
        if args.random_pool == "all":
            if correct:
                arms.append((f"random_correct_s{s}", draw(correct, args.k_demos, seed)))
        else:
            if len(compliant) >= 1:
                arms.append((f"random_compliant_s{s}",
                             draw(compliant, args.k_demos, seed)))
            if len(noncompliant) >= 1:
                arms.append((f"random_noncompliant_s{s}",
                             draw(noncompliant, args.k_demos, seed)))

    report = {"model": args.model, "engine": args.engine, "seed": args.seed,
              "max_tokens": args.max_tokens, "k_demos": args.k_demos,
              "n_seeds": args.n_seeds, "eval_set": args.eval_set,
              "cap_nonnative": args.cap_nonnative, "random_pool": args.random_pool,
              "n_test": len(test),
              "pool": {"correct": len(correct), "compliant": len(compliant),
                       "noncompliant": len(noncompliant)},
              "arms": {}, "conditions": {}}
    all_items = []

    for name, demos in arms:
        solver = CoTSolver()
        solver.predict.demos = demos
        recs = instrumented_eval(test, solver, factory, args.workers, name,
                                 max_tokens=args.max_tokens)
        items = build_items(test, recs, args.model, args.engine, name, args.max_tokens)
        all_items.extend(items)
        payload = demo_payload(demos)
        report["arms"][name] = {
            **payload,
            "subjects": [demo_field(d, "subject", "?") for d in demos]}
        chars = payload["reasoning_chars"]
        report["conditions"][name] = {
            "deployment": score_items(items, "is_correct"),
            "rescue": score_items(items, "rescue_correct")}
        d = report["conditions"][name]["deployment"]
        ona = d["by_subject"].get("ona_tili", {}).get("accuracy")
        trunc = sum(i["truncated"] for i in items)
        errs = sum(i["parse_error"] for i in items)
        print(f"  {name:26} demos={len(demos):2} payload={sum(chars):5} "
              f"ona_tili={ona} overall={d['overall']} trunc={trunc} parse_err={errs}")

    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    (out / f"{safe}_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()
