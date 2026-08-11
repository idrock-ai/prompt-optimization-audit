#!/bin/bash
# E6 driver (runs ON spark-3): serve each large model at bf16 with the box's vllm-env,
# run cot+bootstrap via src.run --engine openai, then restore the pre-existing server.
# Fallback chain per spec: vLLM bf16 -> Ollama Q8 GGUF (noted in results/e6/NOTES.md).
set -uo pipefail
cd ~/prompt-optimization-audit
# ~/.cache/huggingface is root-owned on spark-3 (stale root-run artifact); use our own cache.
export HF_HOME="$HOME/hf_home"
mkdir -p "$HF_HOME" results/e6
NOTES=results/e6/NOTES.md
echo "# E6 serving notes ($(date -u +%FT%TZ))" > "$NOTES"

VLLM=~/vllm-env/bin/python
APERTUS_CMD=$(pgrep -af "vllm.entrypoints.openai.api_server" | grep -v grep | head -1 | cut -d" " -f2-)
APERTUS_PID=$(pgrep -f "vllm.entrypoints.openai.api_server" | head -1 || true)
if [ -n "${APERTUS_PID}" ]; then
  echo "stopping pre-existing vLLM (pid ${APERTUS_PID}): ${APERTUS_CMD}" >> "$NOTES"
  kill "${APERTUS_PID}" || true
  sleep 10
fi

serve_wait() {  # $1 = hf model name
  for i in $(seq 1 120); do
    if curl -sf http://localhost:8000/v1/models 2>/dev/null | grep -q "$1"; then return 0; fi
    sleep 15
  done
  return 1
}

run_model() {  # $1 = HF name, $2 = ollama q8 fallback tag, $3 = safe label
  echo "== $1" >> "$NOTES"
  nohup "$VLLM" -m vllm.entrypoints.openai.api_server --model "$1" \
    --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.80 --enforce-eager \
    > "e6_serve_$3.log" 2>&1 &
  SPID=$!
  if serve_wait "$1"; then
    echo "serving $1 at bf16 (vLLM, pid $SPID)" >> "$NOTES"
    .venv/bin/python -m src.run --model "$1" --engine openai \
      --api-base http://localhost:8000/v1 --conditions cot,bootstrap \
      --max-tokens 512 --out-dir results/e6 >> "e6_run_$3.log" 2>&1 \
      && echo "run OK (bf16)" >> "$NOTES" || echo "run FAILED (bf16) - see e6_run_$3.log" >> "$NOTES"
  else
    echo "vLLM bf16 serve FAILED for $1 (see e6_serve_$3.log) - ABORTING driver (failed loads leak device memory; do not cascade)" >> "$NOTES"
    kill "$SPID" 2>/dev/null; sleep 5
    echo E6_ABORTED > e6.done
    exit 1
  fi
  # graceful teardown: single SIGTERM, wait for full exit, verify memory recovery.
  kill "$SPID" 2>/dev/null
  for i in $(seq 1 36); do
    pgrep -f "[v]llm.entrypoints" >/dev/null || break
    sleep 5
  done
  sleep 15
  AVAIL=$(free -g | awk 'NR==2{print $7}')
  echo "post-teardown available memory: ${AVAIL}GB" >> "$NOTES"
  if [ "${AVAIL:-0}" -lt 100 ]; then
    echo "device memory NOT recovered after teardown - aborting (next load would fail); reboot needed" >> "$NOTES"
    echo E6_ABORTED > e6.done
    exit 1
  fi
}

# Qwen/Qwen3.5-27B: completed at bf16 in the 2026-07-28 morning run (items synced)
# Qwen/Qwen3.6-27B: completed at bf16 (items synced)
run_model "google/gemma-4-31b-it" "gemma4:31b-q8_0" "gemma4_31b"

if [ -n "${APERTUS_CMD:-}" ]; then
  echo "restoring pre-existing server: ${APERTUS_CMD}" >> "$NOTES"
  cd ~ && nohup ${APERTUS_CMD} > ~/apertus_restored.log 2>&1 &
  cd ~/prompt-optimization-audit
fi
echo E6_DONE > e6.done
