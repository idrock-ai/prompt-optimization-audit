"""E9: powered CROSS-subject arm on the frozen replication set.

Why this experiment exists. Our powered design (E4) covers only the WITHIN-subject
regime -- the one where we predicted, and found, no harm. Every cross-subject number in
the paper comes from observational cells of n~100 per model that we ourselves show are
unstable across stacks, precisions and even sessions. This run points the powered
instrument at the regime that actually produces the harm.

Protocol, deliberately identical to E1 except for the evaluation set:
  - demonstrations are bootstrapped from the MIXED-subject benchmark train split, the
    default deployment regime (E4 drew them from ona_tili train only);
  - by default we LOAD E1's saved compiled program rather than recompiling, so the
    demonstrations under test are byte-identical to the ones E1 measured. Recompiling
    would risk drawing a different demo set on the two models that are not run-to-run
    deterministic, which would confound "powered" with "different demos";
  - evaluation is the frozen four-subject public replication set, never used in any
    development decision.

That yields, per model, ~393 native pairs for the absolute McNemar and ~1,635
non-native pairs for the differential odds ratio -- against 100 and 151 in E1.

Usage: python -m src.powered --model qwen3.5:9b --out-dir results/e9
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

from .data import load_splits, cap_per_subject, replication_all
from .program import CoTSolver, metric, compliant_metric, demo_payload
from .instrument import instrumented_eval
from .run import make_lm_factory, build_items, score_items

REPO = Path(__file__).resolve().parent.parent


def load_or_compile(compiled_path: Path | None, trainset, metric_fn, label):
    """Prefer E1's saved program; compile only if it is absent.

    Returns (program, provenance) so the report records which happened -- a reader must
    be able to tell whether the demonstrations were the ones E1 measured or a fresh draw.
    """
    if compiled_path and compiled_path.is_file():
        prog = CoTSolver()
        prog.load(str(compiled_path))
        n = len(getattr(prog.predict, "demos", []))
        print(f"  [{label}] loaded {compiled_path.name} ({n} demos, "
              f"{demo_payload(prog.predict.demos)['reasoning_total']} reasoning chars)")
        return prog, {"source": "loaded", "path": str(compiled_path), "n_demos": n}
    print(f"  [{label}] no saved program at {compiled_path}; compiling fresh")
    prog = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=4,
                            max_labeled_demos=0).compile(CoTSolver(), trainset=trainset)
    n = len(getattr(prog.predict, "demos", []))
    return prog, {"source": "compiled", "path": None, "n_demos": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", choices=("ollama", "openai"), default="ollama")
    ap.add_argument("--api-base", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--train-cap", type=int, default=40,
                    help="per-subject cap on the bootstrap train pool (matches E1)")
    ap.add_argument("--cap-nonnative", type=int, default=None,
                    help="cap each NON-native replication subject (native is never "
                         "thinned). Omit to use all 2,028 items.")
    ap.add_argument("--conditions", default="cot,bootstrap,bootstrap_compliant")
    ap.add_argument("--e1-dir", default="results/e1",
                    help="where to look for the compiled programs to reuse")
    ap.add_argument("--limit", type=int, default=None,
                    help="truncate the eval set (smoke-testing only; never for a "
                         "reported run -- it would silently change the endpoint)")
    ap.add_argument("--out-dir", default="results/e9")
    args = ap.parse_args()

    factory = make_lm_factory(args.engine, args.model, args.api_base, args.max_tokens)
    dspy.configure(lm=factory())

    train, _dev, _test = load_splits(seed=args.seed)
    pool = cap_per_subject(train, args.train_cap)
    test = replication_all(cap_nonnative=args.cap_nonnative)
    if args.limit:
        test = test[:args.limit]
        print(f"  !! --limit {args.limit}: SMOKE RUN, not a reportable result")
    by_subject: dict[str, int] = {}
    for ex in test:
        by_subject[ex.subject] = by_subject.get(ex.subject, 0) + 1
    print(f"[{args.model}] train_pool={len(pool)} test={len(test)} {by_subject}")

    safe = args.model.replace("/", "_").replace(":", "_")
    e1 = REPO / args.e1_dir
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    programs = {}
    if "cot" in conds:
        programs["cot"] = (CoTSolver(), {"source": "zero-shot", "n_demos": 0})
    if "bootstrap" in conds:
        programs["dspy_bootstrap"] = load_or_compile(
            e1 / f"{safe}_bootstrap.json", pool, metric, "vanilla")
    if "bootstrap_compliant" in conds:
        programs["dspy_bootstrap_compliant"] = load_or_compile(
            e1 / f"{safe}_bootstrap_compliant.json", pool, compliant_metric, "compliant")

    report = {"model": args.model, "engine": args.engine, "seed": args.seed,
              "max_tokens": args.max_tokens, "train_cap": args.train_cap,
              "cap_nonnative": args.cap_nonnative, "limit": args.limit,
              "smoke": bool(args.limit), "test_by_subject": by_subject,
              "programs": {}, "conditions": {}}
    all_items = []

    for cond, (prog, provenance) in programs.items():
        report["programs"][cond] = {**provenance,
                                    **demo_payload(getattr(prog.predict, "demos", []))}
        recs = instrumented_eval(test, prog, factory, args.workers, cond,
                                 max_tokens=args.max_tokens)
        items = build_items(test, recs, args.model, args.engine, cond, args.max_tokens)
        all_items.extend(items)
        report["conditions"][cond] = {
            "deployment": score_items(items, "is_correct"),
            "rescue": score_items(items, "rescue_correct")}
        d = report["conditions"][cond]["deployment"]
        errs = sum(i["parse_error"] for i in items)
        trunc = sum(i["truncated"] for i in items)
        native = d["by_subject"].get("ona_tili", {}).get("accuracy")
        print(f"  {cond:26} overall={d['overall']} ona_tili={native} "
              f"parse_errors={errs} truncated={trunc}")

    (out / f"{safe}_items.jsonl").write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in all_items))
    (out / f"{safe}_report.json").write_text(json.dumps(report, indent=1))
    print(f"saved -> {out}/{safe}_*")


if __name__ == "__main__":
    main()
