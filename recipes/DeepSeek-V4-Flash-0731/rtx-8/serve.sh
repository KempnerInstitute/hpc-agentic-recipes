#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash-0731 on one RTX PRO 6000 Blackwell node: FP8 with FP4 experts, TP8.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/DeepSeek-V4-Flash-0731}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }

EXTRA=()
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
# Leave SPEC_MODE unset. The sm120 sparse-MLA kernel needs more than 64 tokens per batch and a draft batch
# is only num_speculative_tokens wide, so both methods abort during warmup. See Known limits.
[ -n "${SPEC_MODE:-}" ] && EXTRA+=(--speculative-config \
  "{\"method\": \"${SPEC_MODE}\", \"num_speculative_tokens\": ${SPEC_TOKENS:-1}}")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name deepseek-v4-flash \
  --kv-cache-dtype fp8_ds_mla \
  --tensor-parallel-size "${TP:-8}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-1048576}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-deepseek_v4}" \
  --reasoning-parser "${REASONING_PARSER:-deepseek_v4}" \
  "${EXTRA[@]}"
