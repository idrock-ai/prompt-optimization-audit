#!/bin/bash
# E11: power the substitution finding on the frozen replication set.
#
# E10 showed, on 4 models x 251 DTM test items, that BootstrapFewShot's selected
# demonstrations do WORSE than random draws from the same pool of correct examples
# (p=0.0097) -- and that requiring brevity-compliance is inert (p=0.90). That is now the
# paper's candidate headline, and it rests on exactly the cell size that produced two
# claims E9 has already overturned. So it gets powered before it gets written.
#
# Design:
#   - eval on the frozen four-subject public corpus, --cap-nonnative 100
#     (native keeps all 393; non-native 100 each) = 693 items x 6 models
#   - arms: vanilla (BSFS's own picks) + 2 random draws from the WHOLE correct pool.
#     Compliance is already settled as null, so the split is dropped and the budget
#     goes into items instead.
#   - CoT is NOT re-run: E9 already has it on a superset of these items (cap 150
#     contains cap 100 under the same seeded shuffle). The PRIMARY contrast --
#     vanilla vs random -- is therefore fully within-session and clean; only the
#     secondary "random vs no-demos" comparison inherits E9's session gap, and only
#     for the two gemma models that are not run-to-run deterministic.
set -euo pipefail
OUT="${OUT_DIR:-results/e11}"
SEEDS="${N_SEEDS:-2}"
CAP="${CAP_NONNATIVE:-100}"

for m in qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.5:27b qwen3.6:27b gemma4:31b; do
  safe="${m//[:\/]/_}"
  if [ -f "$OUT/${safe}_items.jsonl" ]; then
    echo "== $m: already done, skipping"; continue
  fi
  echo "== $m"
  .venv/bin/python -m src.substitution --model "$m" --engine ollama \
    --eval-set replication --cap-nonnative "$CAP" \
    --random-pool all --arms vanilla --n-seeds "$SEEDS" \
    --max-tokens 512 --out-dir "$OUT"
done
echo "E11 complete -> $OUT"
