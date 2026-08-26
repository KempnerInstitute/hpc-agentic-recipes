#!/usr/bin/env bash
# Serve Qwen3.8-27B-FP8 on one H200 GPU: TP1, FP8 weights, hybrid attention with a vision tower.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Qwen3.8-27B-FP8}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }

EXTRA=()
if [ -n "${ENFORCE_EAGER:-}" ]; then
  EXTRA+=(--enforce-eager)
else
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\"}")
fi
[ "${MTP_TOKENS:-3}" != 0 ] && EXTRA+=(--speculative-config \
  "{\"method\": \"${SPEC_METHOD:-mtp}\", \"num_speculative_tokens\": ${MTP_TOKENS:-3}}")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-fp8 \
  --tensor-parallel-size "${TP:-1}" \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${EXTRA[@]}"
