#!/usr/bin/env bash
# Serve Qwen3.8-2.4T-A95B-FP8 across six H200 nodes: TP4 inside each node, PP6 between them, over Ray.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Qwen3.8-2.4T-A95B-FP8}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }
HEAD_IP="${1:-${RAY_HEAD_IP:-}}"
[ -n "$HEAD_IP" ] || { echo "usage: $(basename "$0") <head_ip> [ray_port]   (or set RAY_HEAD_IP)" >&2; exit 2; }
export RAY_ADDRESS="$HEAD_IP:${2:-${RAY_PORT:-6379}}"

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
  --served-model-name qwen3.8-2.4t \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-6}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --load-format "${LOAD_FORMAT:-fastsafetensors}" \
  --safetensors-load-strategy "${LOAD_STRATEGY:-lazy}" \
  --kv-cache-dtype "${KV_DTYPE:-fp8}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${EXTRA[@]}"
