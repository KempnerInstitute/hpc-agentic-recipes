#!/usr/bin/env bash
# Serve GLM-5.3 across two H200 nodes: TP4 inside each node, PP2 between them, over Ray.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/GLM-5.3}"
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
[ "${MTP_TOKENS:-0}" != 0 ] && EXTRA+=(--speculative-config \
  "{\"method\": \"${SPEC_METHOD:-mtp}\", \"num_speculative_tokens\": ${MTP_TOKENS:-0}}")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name glm-5.3 \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-2}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-641664}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.94}" \
  --enable-auto-tool-choice \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  --tool-call-parser "${TOOL_PARSER:-glm47}" \
  --reasoning-parser "${REASONING_PARSER:-glm47}" \
  "${EXTRA[@]}"
