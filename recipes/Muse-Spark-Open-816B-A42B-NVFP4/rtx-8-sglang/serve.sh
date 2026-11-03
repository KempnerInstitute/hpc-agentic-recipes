#!/usr/bin/env bash
# Serve Muse-Spark-Open-816B-A42B-NVFP4 on one RTX PRO 6000 Blackwell node under SGLang, in a container.
# TP8 with expert parallelism and DFlash speculative decoding.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Muse-Spark-Open-816B-A42B-NVFP4-rtx-8-sglang}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4}"
DRAFT="${DRAFT:-$MODELS_DIR/Muse-Spark-Open-DFlash2}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

mkdir -p "$MUSE_LOG_DIR/hf" "$MUSE_LOG_DIR/triton" "$MUSE_LOG_DIR/inductor"
LAUNCHER="$REPO_ROOT/common/tools/sglang_launch.py"
if [ -n "${VLLM_API_KEY:-}" ]; then
  export SINGULARITYENV_VLLM_API_KEY="$VLLM_API_KEY"
  export APPTAINERENV_VLLM_API_KEY="$VLLM_API_KEY"
fi

SPEC=()
BINDS=(-B "$MODEL:$MODEL:ro" -B "$MUSE_LOG_DIR:$MUSE_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro")
if [ "${NO_SPEC:-0}" != 1 ] && [ -d "$DRAFT" ]; then
  BINDS+=(-B "$DRAFT:$DRAFT:ro")
  SPEC=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFT")
fi

LOG="$MUSE_LOG_DIR/muse-nvfp4-sglang.log"
{
  echo "=== $(date '+%H:%M:%S') on $(hostname -s), context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
  echo "=== spec ${SPEC[*]:-none}"
} >> "$LOG"
exec >> "$LOG" 2>&1

exec singularity exec --nv \
  "${BINDS[@]}" \
  --env NCCL_P2P_DISABLE="$NCCL_P2P_DISABLE" \
  --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env HF_HOME="$HF_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$SIF" \
  python3 "$LAUNCHER" \
    --model-path "$MODEL" \
    --served-model-name muse-spark \
    --tp-size "${TP:-8}" \
    --ep-size "${EP:-8}" \
    --moe-runner-backend "${MOE_RUNNER:-flashinfer_trtllm}" \
    --attention-backend "${ATTN:-trtllm_mha}" \
    --kv-cache-dtype "${KV_DTYPE:-bfloat16}" \
    --context-length "$MAX_MODEL_LEN" \
    --page-size "${PAGE_SIZE:-128}" \
    --mem-fraction-static "$MEM_FRACTION" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-4096}" \
    --reasoning-parser muse \
    --tool-call-parser muse \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    "${SPEC[@]}" ${EXTRA_ARGS:-}
