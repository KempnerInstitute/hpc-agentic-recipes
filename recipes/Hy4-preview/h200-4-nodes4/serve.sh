#!/usr/bin/env bash
# Serve Hy4-preview across four H200 nodes under SGLang: pure TP16, in a container.
# The model rejects pipeline parallelism, so TP is the only axis, and TP must be a power of two
# dividing the 64 attention heads.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Hy4-preview-h200-4-nodes4}"
source "$REPO_ROOT/common/lib/api_key.sh"
[ -n "$(printf '%s' "${VLLM_API_KEY:-}" | tr -d '[:space:]')" ] || {
  echo "no API key resolved: this endpoint would accept unauthenticated requests. Create" >&2
  echo "secrets/$KEY_NAME.key or set VLLM_API_KEY, then retry." >&2
  exit 1
}
MODEL="${MODEL:-$MODELS_DIR/Hy4-preview}"
RANK="${RANK:?set RANK, 0 through 3}"
HEAD_IB="${HEAD_IB:?set HEAD_IB, the head node ib0 address}"
API_PORT="${API_PORT:-8000}"
# The checkpoint declares 1048576, but the cookbook says that exceeds the per-rank pool on H200.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
mkdir -p "$HY4_LOG_DIR/hf" "$HY4_LOG_DIR/triton" "$HY4_LOG_DIR/inductor"
# SGLang logs the parsed server_args, the API key among them, so the log must not be world readable.
chmod 700 "$HY4_LOG_DIR" || { echo "cannot restrict $HY4_LOG_DIR; refusing to log the API key into a readable file" >&2; exit 1; }
LOG="$HY4_LOG_DIR/hy4-rank$RANK.log"
umask 077
: > "$LOG" || { echo "cannot create $LOG" >&2; exit 1; }
chmod 600 "$LOG" || { echo "cannot restrict $LOG; refusing to log the API key into a readable file" >&2; exit 1; }
# This rank's own pid, so serve_ssh.sh checks this process rather than any same-user launcher.
echo $$ > "$HY4_LOG_DIR/hy4-rank$RANK.pid"

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

# No MoE backend is pinned: the cookbook pins none for this model.
MOE=()
[ -n "${MOE_RUNNER:-}" ] && MOE=(--moe-runner-backend "$MOE_RUNNER")
# The native MTP layer is driven by NEXTN here, not by the mtp flags other recipes use. Off by default
# for a first bring-up; the cookbook's low-latency preset turns it on.
SPEC=()
if [ "${SPEC_MODE:-none}" = nextn ]; then
  SPEC=(--speculative-algorithm NEXTN
        --speculative-num-steps "${SPEC_STEPS:-3}"
        --speculative-eagle-topk "${SPEC_TOPK:-1}"
        --speculative-num-draft-tokens "${SPEC_DRAFT:-4}")
fi

# Do not override the attention backend without a kernel-specific reason: SGLang auto-selects DSA with
# flashmla_sparse for this model, over an FP8 indexer cache and a bf16 KV pool.
ATTNA=()
[ -n "${ATTN:-}" ] && ATTNA=(--attention-backend "$ATTN")

{
  echo "=== rank $RANK, $(date '+%H:%M:%S') on $(hostname -s)"
  echo "=== head $HEAD_IB:$DIST_PORT, hca $_hca, context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
  echo "=== moe ${MOE_RUNNER:-auto}, attn ${ATTN:-auto}"
} >> "$LOG"
exec >> "$LOG" 2>&1

# SGLang hardcodes an unbalanced-load barrier at 480 s, which this checkpoint cannot meet: 16 ranks
# loading 1452.8 GiB from shared storage spread their finish times by about 590 s in two measured
# attempts, so correctly loading ranks were killed. There is no environment variable for it, so the
# file is bind-mounted with a larger value. See patches/README.md.
BINDS=(-B "$MODEL:$MODEL:ro" -B "$HY4_LOG_DIR:$HY4_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro")
if [ "${HY4_LOAD_BARRIER_PATCH:-1}" = 1 ]; then
  _pp="$S/patches/load_model_utils.py"
  _pp_dst=/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner_components/load_model_utils.py
  [ -f "$_pp" ] || { echo "HY4_LOAD_BARRIER_PATCH=1 but $_pp is missing" >&2; exit 1; }
  BINDS+=(-B "$_pp:$_pp_dst:ro")
fi

exec singularity exec --nv \
  "${BINDS[@]}" \
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
    --served-model-name hy4-preview \
    --trust-remote-code \
    --tool-call-parser "${TOOL_PARSER:-auto}" \
    --reasoning-parser "${REASONING_PARSER:-auto}" \
    --tp-size "${TP:-16}" \
    --nnodes "${NNODES:-4}" \
    --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IB:$DIST_PORT" \
    --host 0.0.0.0 --port "$API_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --weight-loader-disable-mmap \
    "${SPEC[@]}" "${MOE[@]}" "${ATTNA[@]}" ${EXTRA_ARGS:-}
