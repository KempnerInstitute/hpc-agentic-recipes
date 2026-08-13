#!/usr/bin/env bash
# Serve Kimi-K2.7-Code across two H200 nodes: TP4 inside each node, PP2 between them, over Ray.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K2.7-Code}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }
HEAD_IP="${1:-${RAY_HEAD_IP:-}}"
[ -n "$HEAD_IP" ] || { echo "usage: $(basename "$0") <head_ip> [ray_port]   (or set RAY_HEAD_IP)" >&2; exit 2; }
export RAY_ADDRESS="$HEAD_IP:${2:-${RAY_PORT:-6379}}"

EXTRA=()
# Graph capture needs the allreduce and RMSNorm fusion pass off across nodes, or it takes an illegal
# memory access.
if [ -n "${ENFORCE_EAGER:-}" ]; then
  EXTRA+=(--enforce-eager)
else
  EXTRA+=(--compilation-config \
    "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}\", \"pass_config\": {\"fuse_allreduce_rms\": false}}")
fi
[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA+=(${EXTRA_ARGS:---skip-mm-profiling --mm-processor-cache-gb 0})

exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.7-code \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-2}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-262144}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-kimi_k2}" \
  --reasoning-parser "${REASONING_PARSER:-kimi_k2}" \
  "${EXTRA[@]}"
