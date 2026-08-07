#!/usr/bin/env bash
# Serve GLM-4.6-FP8 on one H200 node: TP4, eager, with MTP speculative decoding.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/GLM-4.6-FP8}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=()
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
[ -z "${NO_MTP:-}" ] && EXTRA+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS:-1}}")

exec vllm serve "$MODEL" \
  --served-model-name glm-4.6 \
  --tensor-parallel-size "${TP:-4}" \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-202752}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm45}" \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  "${EXTRA[@]}"
