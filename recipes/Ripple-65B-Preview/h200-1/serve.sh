#!/usr/bin/env bash
# Serve Ripple-65B-Preview on one H200 GPU under SGLang, in a container, using the SGLang model
# implementation and launcher that ship inside the checkpoint.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Ripple-65B-Preview-h200-1}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Ripple-65B-Preview}"
DRAFT="${DRAFT:-$MODELS_DIR/Ripple-65B-Preview-DFlash2}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MEM_FRACTION="${MEM_FRACTION:-0.93}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
SERVE="$MODEL/inference/server/model.py"
[ -f "$SERVE" ] || { echo "the checkpoint model implementation is missing: $SERVE" >&2; exit 1; }
mkdir -p "$RIPPLE_LOG_DIR/hf" "$RIPPLE_LOG_DIR/triton" "$RIPPLE_LOG_DIR/inductor"
# SGLang logs the parsed server_args, the API key among them, so the log must not be world readable.
# The default /tmp location is on a shared node and would otherwise be mode 644 under mode 755.
chmod 700 "$RIPPLE_LOG_DIR" 2>/dev/null || true

LAUNCHER="$REPO_ROOT/common/tools/ripple_launch.py"
if [ -n "${VLLM_API_KEY:-}" ]; then
  export SINGULARITYENV_VLLM_API_KEY="$VLLM_API_KEY"
  export APPTAINERENV_VLLM_API_KEY="$VLLM_API_KEY"
fi

# Speculative decoding is opt-in rather than on by default: the DFlash path the checkpoint ships
# fails on H200, see the recipe README. Setting all of the draft flags together matters, because a
# partial speculative configuration makes the tokenizer manager fail on the first request.
SPEC=()
if [ "${SPEC_MODE:-none}" = dflash ]; then
  [ -d "$DRAFT" ] || { echo "SPEC_MODE=dflash needs the draft checkpoint at $DRAFT" >&2; exit 1; }
  SPEC=(--speculative-algorithm DFLASH
        --speculative-draft-model-path "$DRAFT"
        --speculative-draft-model-quantization unquant
        --speculative-num-draft-tokens "${SPEC_TOKENS:-16}"
        --speculative-draft-window-size "${SPEC_WINDOW:-16384}"
        --speculative-draft-attention-backend triton)
fi

# The draft is bound into the container only when speculation is on, so a missing draft checkpoint
# does not stop the server from starting without it.
DRAFT_BIND=()
[ "${SPEC_MODE:-none}" = dflash ] && DRAFT_BIND=(-B "$DRAFT:$DRAFT:ro")

LOG="$RIPPLE_LOG_DIR/ripple-sglang.log"
umask 077
: > "$LOG" 2>/dev/null || true
chmod 600 "$LOG" 2>/dev/null || true
{
  echo "=== $(date '+%H:%M:%S') on $(hostname -s), context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION, tp ${TP:-1}"
  echo "=== spec ${SPEC[*]}"
} >> "$LOG"
exec >> "$LOG" 2>&1

# Flags here are appended to the checkpoint launcher's own defaults, so they override them. The
# default host is 127.0.0.1, which would make the endpoint unreachable from other nodes.
exec singularity exec --nv \
  -B "$MODEL:$MODEL:ro" "${DRAFT_BIND[@]}" \
  -B "$RIPPLE_LOG_DIR:$RIPPLE_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro" \
  --env SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN="$SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN" \
  --env RIPPLE_SERVER_DIR="$MODEL/inference/server" \
  --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" --env HF_HOME="$HF_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  "$SIF" \
  python3 "$LAUNCHER" \
    --model-path "$MODEL" \
    --served-model-name ripple \
    --trust-remote-code \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --tp-size "${TP:-1}" \
    --attention-backend "${ATTN:-fa3}" --page-size "${PAGE_SIZE:-64}" \
    --host 0.0.0.0 --port "$API_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --max-running-requests "${MAX_RUNNING:-8}" \
    --cuda-graph-backend-prefill disabled \
    --tool-call-parser ripple --reasoning-parser ripple \
    --chat-template "$MODEL/chat_template.jinja" \
    "${SPEC[@]}" ${EXTRA_ARGS:-}
