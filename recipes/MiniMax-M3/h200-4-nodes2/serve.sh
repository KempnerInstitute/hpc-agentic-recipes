#!/usr/bin/env bash
# Serve MiniMax-M3 across two H200 nodes: pure TP8 over Ray, BF16: TP4 inside each node, PP2 between them, over Ray.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/MiniMax-M3}"
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
# Upstream uses EAGLE3 for this model, not MTP, and no draft checkpoint is downloaded here, so
# speculation stays off. Set SPEC_TOKENS and SPEC_MODEL together if you add a draft.
[ "${SPEC_TOKENS:-${MTP_TOKENS:-0}}" != 0 ] && EXTRA+=(--speculative-config \
  "{\"method\": \"${SPEC_METHOD:-eagle3}\", \"num_speculative_tokens\": ${SPEC_TOKENS:-${MTP_TOKENS:-0}}}")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

# --block-size 128 is mandatory on every platform for this model. The default 16 misaligns the
# sparse attention, and upstream reports it failing outright on some backends. Do not drop it.
exec vllm serve "$MODEL" \
  --block-size "${BLOCK_SIZE:-128}" \
  --served-model-name minimax-m3 \
  --tensor-parallel-size "${TP:-8}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.94}" \
  --enable-auto-tool-choice \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  --tool-call-parser "${TOOL_PARSER:-minimax_m3}" \
  --reasoning-parser "${REASONING_PARSER:-minimax_m3}" \
  "${EXTRA[@]}"
