#!/usr/bin/env bash
# Serve Qwen3-Coder-480B-A35B-Instruct-FP8 on one RTX PRO 6000 Blackwell node: TP4 x PP2, FP8, CUDA
# graphs, prefix caching.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-480B-A35B-Instruct-FP8}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=(--pipeline-parallel-size "${PP:-2}" --distributed-executor-backend "${EXECUTOR:-mp}")
_RP="${REASONING_PARSER-}"
[ -n "$_RP" ] && EXTRA+=(--reasoning-parser "$_RP")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name qwen3-coder-480b \
  --tensor-parallel-size "${TP:-4}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  "${EXTRA[@]}"
