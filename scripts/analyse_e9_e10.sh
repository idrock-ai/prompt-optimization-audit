#!/bin/bash
# Every analysis the two new sweeps feed, in one pass. Safe to run on partial results:
# each analysis simply reports whatever models have landed so far.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

if ls results/e9/*_items.jsonl >/dev/null 2>&1; then
  echo "############ E9: powered cross-subject arm ############"
  echo "--- absolute native McNemar + format share (the PRE-SPECIFIED endpoint)"
  $PY analysis/decompose.py   results/e9 --subject ona_tili
  echo; echo "--- differential odds ratio, powered"
  $PY analysis/interaction.py results/e9 --native ona_tili
  echo; echo "--- placebo rotation, powered"
  $PY analysis/placebo.py     results/e9 --native ona_tili
else
  echo "E9: no results yet"
fi

if ls results/e10/*_items.jsonl >/dev/null 2>&1; then
  echo; echo "############ E10: constraint vs substitution ############"
  $PY analysis/substitution_stats.py results/e10 --subject ona_tili
else
  echo; echo "E10: no results yet"
fi

$PY analysis/paper_numbers.py
