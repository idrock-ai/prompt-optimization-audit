#!/usr/bin/env python3
"""Export every paper-facing number to results/paper_numbers.json: the single
source for every in-text number in the paper.

Scope: e1 (flip decomposition + ladder), e2 (budget dose-response, non-degenerate
models only), e3 (demo length x subject paired contrasts), e4 (powered
within-subject residual, paper + replication sets), e5 (fixes shoot-out, merged
verbatim from results/e5/fixes.json), e7 (TurkishMMLU), e9 (powered CROSS-subject
arm), e10 (demonstration-substitution ablation), plus the differential/placebo/
determinism analyses.

Safe to run at ANY pipeline stage -- every section is guarded by an existence
check on the directory/file it reads, so an experiment that hasn't produced
results yet -- or has produced only degenerate/partial data -- simply leaves
its key absent from the output. Never crashes, never fabricates a value for a
sweep that hasn't finished.

Usage: python analysis/paper_numbers.py [root]   (root defaults to ".")
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, ".")
from analysis.decompose import decompose_dir
from analysis.dose_response import collect


def _load_json(path):
    """Return the parsed JSON at `path`, or None if it doesn't exist yet.

    Returns None on malformed JSON, with a warning to stderr.
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[paper_numbers] WARNING: skipping malformed {path}: {e}", file=sys.stderr)
        return None


def build(root="."):
    """Collect every paper-facing number found under `root` into one dict.

    Every key is independently guarded by an existence check on the directory
    or file it reads, so this never raises for a partially-complete results/
    tree -- it just omits whatever isn't ready yet.
    """
    out = {}

    e1_dir = os.path.join(root, "results", "e1")
    if os.path.isdir(e1_dir):
        out["e1"] = decompose_dir(e1_dir)

    e2_dir = os.path.join(root, "results", "e2")
    has_e2_data = any(glob.glob(os.path.join(e2_dir, f"mt{b}", "*_items.jsonl"))
                      for b in (256, 1024, 2048))
    if has_e2_data:
        e2 = {}
        for m, row in collect(e2_dir, e1_dir=e1_dir).items():
            if sum(1 for k in row if isinstance(k, int)) < 2:
                continue  # e1-fallback degenerate row: only the borrowed 512
                          # cell, no real per-budget sweep for this model
            e2[m] = {str(k): v for k, v in row.items()}
        if e2:
            out["e2"] = e2

    e3_dir = os.path.join(root, "results", "e3")
    if glob.glob(os.path.join(e3_dir, "*_items.jsonl")):
        from analysis.demolab_stats import collect_stats
        out["e3"] = collect_stats(e3_dir)

    e4_dir = os.path.join(root, "results", "e4")
    if glob.glob(os.path.join(e4_dir, "*_paper_items.jsonl")):
        from analysis.residual_stats import collect_set
        out["e4"] = {"paper": collect_set(e4_dir, "paper"),
                     "replication": collect_set(e4_dir, "replication")}

    # Raw JSON already serialized by E5's fixes shoot-out script -- merged in
    # verbatim when present (optional: the pipeline may not have reached E5 yet).
    e5_fixes = _load_json(os.path.join(root, "results", "e5", "fixes.json"))
    if e5_fixes is not None:
        out["e5_fixes"] = e5_fixes

    # Turkish (E7) and MIPROv2 (E8) flip decompositions, so the format-share number
    # that bounds the mechanism's reach is exported alongside the DTM one.
    e7_dir = os.path.join(root, "results", "e7")
    if glob.glob(os.path.join(e7_dir, "*_items.jsonl")):
        out["e7"] = decompose_dir(e7_dir, subject="Turkish_Language_and_Literature")

    # Differential-harm estimates and their placebo rotations, keyed by the results
    # directory they came from. Both are written by their own scripts; merged verbatim.
    for key, d, fname in (("interaction_original", "main", "interaction.json"),
                          ("interaction_replication", "e1", "interaction.json"),
                          ("interaction_turkish", "e7", "interaction.json"),
                          ("interaction_mipro", "e8", "interaction.json"),
                          ("placebo_original", "main", "placebo.json"),
                          ("placebo_replication", "e1", "placebo.json"),
                          ("placebo_turkish", "e7", "placebo.json")):
        blob = _load_json(os.path.join(root, "results", d, fname))
        if blob is not None:
            out[key] = blob

    # E9 (powered cross-subject) and E10 (substitution ablation). Both are guarded by
    # existence checks like every other section, so this export works before, during and
    # after the sweeps.
    e9_dir = os.path.join(root, "results", "e9")
    if glob.glob(os.path.join(e9_dir, "*_items.jsonl")):
        out["e9"] = decompose_dir(e9_dir)
    for key, d, fname in (("interaction_powered", "e9", "interaction.json"),
                          # the subject-TYPE contrast: the paper's headline claim as a
                          # single estimate, rather than two marginal tests read together
                          ("subject_type_powered", "e9", "subject_type.json"),
                          ("subject_type_original", "main", "subject_type.json"),
                          ("placebo_powered", "e9", "placebo.json"),
                          ("e10_substitution", "e10", "substitution_stats.json")):
        blob = _load_json(os.path.join(root, "results", d, fname))
        if blob is not None:
            out[key] = blob

    determinism = _load_json(os.path.join(root, "results", "determinism.json"))
    if determinism is not None:
        out["determinism"] = determinism

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    a = ap.parse_args()
    out = build(a.root)
    out_path = os.path.join(a.root, "results", "paper_numbers.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1, default=str)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
