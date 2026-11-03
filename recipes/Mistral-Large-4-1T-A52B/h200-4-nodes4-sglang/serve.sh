#!/usr/bin/env bash
# Serve Mistral-Large-4-1T-A52B FP8 across four H200 nodes under SGLang: TP16, in a container.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Mistral-Large-4-1T-A52B-h200-4-nodes4}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Mistral-Large-4-1T-A52B}"
RANK="${RANK:?set RANK, 0 through 3}"
HEAD_IB="${HEAD_IB:?set HEAD_IB, the head node ib0 address}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
mkdir -p "$ML4_LOG_DIR/hf" "$ML4_LOG_DIR/triton" "$ML4_LOG_DIR/inductor"
LOG="$ML4_LOG_DIR/ml4-rank$RANK.log"

_hca=""
for _d in /sys/class/infiniband/*; do
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  case "$_s$_l" in *ACTIVE*InfiniBand*) _hca+="${_hca:+,}$(basename "$_d")" ;; esac
done

LAUNCHER="$REPO_ROOT/common/tools/sglang_launch.py"
if [ -n "${VLLM_API_KEY:-}" ]; then
  export SINGULARITYENV_VLLM_API_KEY="$VLLM_API_KEY"
  export APPTAINERENV_VLLM_API_KEY="$VLLM_API_KEY"
fi

SPEC=()
# DSpark drafting ships inside the checkpoint; SGLang loads it without a draft path.
[ "${NO_SPEC:-0}" = 1 ] && SPEC=(--speculative-algorithm NONE)

{
  echo "=== rank $RANK, $(date '+%H:%M:%S') on $(hostname -s)"
  echo "=== head $HEAD_IB:$DIST_PORT, hca $_hca, context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
} >> "$LOG"
exec >> "$LOG" 2>&1

exec singularity exec --nv \
  -B "$MODEL:$MODEL:ro" -B "$ML4_LOG_DIR:$ML4_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro" \
  --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  --env GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  --env NCCL_IB_HCA="$_hca" \
  --env NCCL_DEBUG=WARN \
  --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" --env HF_HOME="$HF_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$SIF" \
  python3 "$LAUNCHER" \
    --model-path "$MODEL" \
    --served-model-name mistral-large-4 \
    --tp-size "${TP:-16}" \
    --nnodes "${NNODES:-4}" \
    --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IB:$DIST_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "${MAX_RUNNING:-48}" \
    --reasoning-parser mistral \
    --tool-call-parser mistral \
    --host 0.0.0.0 --port "$API_PORT" \
    "${SPEC[@]}" ${EXTRA_ARGS:-}
