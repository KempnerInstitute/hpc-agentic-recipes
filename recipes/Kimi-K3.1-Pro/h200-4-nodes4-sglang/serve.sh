#!/usr/bin/env bash
# Serve Kimi-K3.1-Pro across four H200 nodes under SGLang: TP16 with EP16, in a container.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Kimi-K3.1-Pro-h200-4-nodes4}"
source "$REPO_ROOT/common/lib/api_key.sh"
[ -n "$(printf '%s' "${VLLM_API_KEY:-}" | tr -d '[:space:]')" ] || {
  echo "no API key resolved: this endpoint would accept unauthenticated requests. Create" >&2
  echo "secrets/$KEY_NAME.key or set VLLM_API_KEY, then retry." >&2
  exit 1
}
MODEL="${MODEL:-$MODELS_DIR/Kimi-K3.1-Pro}"
RANK="${RANK:?set RANK, 0 through 3}"
HEAD_IB="${HEAD_IB:?set HEAD_IB, the head node ib0 address}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
mkdir -p "$K31_LOG_DIR/hf" "$K31_LOG_DIR/triton" "$K31_LOG_DIR/inductor"
# SGLang logs the parsed server_args, the API key among them, so the log must not be world readable.
chmod 700 "$K31_LOG_DIR" || { echo "cannot restrict $K31_LOG_DIR; refusing to log the API key into a readable file" >&2; exit 1; }
LOG="$K31_LOG_DIR/k31-rank$RANK.log"
umask 077
: > "$LOG" || { echo "cannot create $LOG" >&2; exit 1; }
chmod 600 "$LOG" || { echo "cannot restrict $LOG; refusing to log the API key into a readable file" >&2; exit 1; }
# This rank's own pid, so serve_ssh.sh checks this process rather than any same-user launcher.
echo $$ > "$K31_LOG_DIR/k31-rank$RANK.pid"

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

# The KV cache dtype is not a free choice here: the engine validates it per model and refuses
# anything but fp8_e4m3, the reference fp8_ds_mla format, even though argparse accepts bf16 and the
# sibling Kimi-K3 recipe serves bf16 happily.
# Marlin is mandatory on Hopper, not a tuning knob: MXFP4 is Blackwell-native, and the auto backend
# dequantizes every expert to BF16 and runs out of memory. The Kimi-K3 recipe pins it for the same
# reason. Override only to experiment.
MOE=(--moe-runner-backend "${MOE_RUNNER:-marlin}")
ATTNA=()
[ -n "${ATTN:-}" ] && ATTNA=(--attention-backend "$ATTN")

{
  echo "=== rank $RANK, $(date '+%H:%M:%S') on $(hostname -s)"
  echo "=== head $HEAD_IB:$DIST_PORT, hca $_hca, context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
  echo "=== moe ${MOE_RUNNER:-auto}, attn ${ATTN:-auto}"
} >> "$LOG"
exec >> "$LOG" 2>&1

exec singularity exec --nv \
  -B "$MODEL:$MODEL:ro" -B "$K31_LOG_DIR:$K31_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro" \
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
    --served-model-name kimi-k3.1-pro \
    --trust-remote-code \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --tp-size "${TP:-16}" \
    --ep-size "${EP:-16}" \
    --nnodes "${NNODES:-4}" \
    --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IB:$DIST_PORT" \
    --host 0.0.0.0 --port "$API_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
    --weight-loader-disable-mmap \
    "${MOE[@]}" "${ATTNA[@]}" ${EXTRA_ARGS:-}
