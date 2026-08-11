#!/bin/bash
# E10: demonstration-substitution ablation -- constraint or substitution?
#
# Separates the two things the compliant metric changes at once: the demonstrations now
# satisfy the brevity instruction, AND they are a different set of examples. Random draws
# from each side of the compliance split, several seeds, against the two arms E1 already
# ran. See src/substitution.py for how to read the result.
#
# Defaults to the four models that actually erode on the replication stack; set
# MODELS to override. Resumable like run_e9.sh.
set -euo pipefail
MODELS="${MODELS:-qwen3.5:4b qwen3.5:9b gemma4:e4b qwen3.6:27b}"
OUT="${OUT_DIR:-results/e10}"
SEEDS="${N_SEEDS:-3}"

for m in $MODELS; do
  safe="${m//[:\/]/_}"
  if [ -f "$OUT/${safe}_items.jsonl" ]; then
    echo "== $m: already done, skipping"
    continue
  fi
  echo "== $m"
  .venv/bin/python -m src.substitution --model "$m" --engine ollama \
    --max-tokens 512 --n-seeds "$SEEDS" --out-dir "$OUT"
done
echo "sweep complete -> $OUT"
