#!/bin/bash
# E9 + E10 driver. ONE GPU, so the two sweeps run SERIALLY, never concurrently.
#
# Sizing rationale (measured, not guessed -- see the calibration in the E9 memo):
#   E9 uses --cap-nonnative 150. The NATIVE subject is never thinned, so the
#   pre-specified primary endpoint keeps all 393 items x 6 models = 2,358 pairs
#   (E1 had 600). Only the comparison stratum is sampled: 450 x 6 = 2,700 non-native
#   pairs (E1 had 906). Using all 2,028 items would cost 56 h instead of 23 h to buy
#   power the primary endpoint does not need.
#   E9 runs cot + vanilla bootstrap only. The compliant arm at scale would be a bonus;
#   the fix's evidence already lives in E1/E5, and adding it here costs +50%.
#
# Both sweeps are resumable: a model whose *_items.jsonl exists is skipped, so a reboot
# or a dropped ssh costs one model, not the run.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/e9e10.log
mkdir -p results
{
  echo "=== driver start $(date -Is)"
  echo "--- E9: powered cross-subject arm (cap-nonnative=150, cot+bootstrap)"
  CAP_NONNATIVE=150 CONDITIONS=cot,bootstrap bash scripts/run_e9.sh
  echo "--- E9 done $(date -Is)"
  echo "--- E10: demonstration-substitution ablation (3 seeds, 4 eroding models)"
  bash scripts/run_e10.sh
  echo "--- E10 done $(date -Is)"
  echo "=== driver complete $(date -Is)"
} >> "$LOG" 2>&1
touch results/e9e10.done
