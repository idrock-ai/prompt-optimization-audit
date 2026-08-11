#!/bin/bash
# E9: powered CROSS-subject arm on the frozen four-subject replication set.
#
# Answers the objection the paper currently concedes in Limitations: our n=3,822 powered
# test (E4) covers only the WITHIN-subject regime, while every cross-subject number comes
# from n~100 observational cells we ourselves show are unstable. This run evaluates the
# demonstrations E1 already measured against ~2,000 held-out items per model.
#
# Resumable: a model whose *_items.jsonl already exists is skipped, so an interrupted
# sweep (or a spark-3 reboot) costs one model, not the run.
#
# Runtime knob: --cap-nonnative caps each NON-native subject; the native subject is never
# thinned, so the primary endpoint keeps its full 393 items either way.
#   (unset)  = 2,028 items/model  ~ full power, longest run
#   250      = 1,143 items/model  ~ good power, roughly 55% of the cost
set -euo pipefail
CAP="${CAP_NONNATIVE:-}"        # e.g. CAP_NONNATIVE=250 scripts/run_e9.sh
OUT="${OUT_DIR:-results/e9}"
CONDS="${CONDITIONS:-cot,bootstrap,bootstrap_compliant}"
CAP_ARG=()
[ -n "$CAP" ] && CAP_ARG=(--cap-nonnative "$CAP")

for m in qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b; do
  safe="${m//[:\/]/_}"
  if [ -f "$OUT/${safe}_items.jsonl" ]; then
    echo "== $m: already done, skipping"
    continue
  fi
  echo "== $m"
  .venv/bin/python -m src.powered --model "$m" --engine ollama \
    --conditions "$CONDS" \
    --max-tokens 512 "${CAP_ARG[@]}" --out-dir "$OUT"
done
echo "sweep complete -> $OUT"
