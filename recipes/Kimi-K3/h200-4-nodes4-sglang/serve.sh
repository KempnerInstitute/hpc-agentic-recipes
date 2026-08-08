#!/usr/bin/env bash
# Serve Kimi-K3 across four H200 nodes under SGLang: TP16 with expert parallelism, in a container.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME=Kimi-K3-h200-4-nodes4-sglang
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K3}"
DRAFT="${DRAFT:-$MODELS_DIR/Kimi-K3-DSpark}"
RANK="${RANK:?set RANK, 0 through 3}"
HEAD_IB="${HEAD_IB:?set HEAD_IB, the head node ib0 address}"
SPEC_MODE="${SPEC_MODE:-none}"
API_PORT="${API_PORT:-8000}"
DIST_PORT="${DIST_PORT:-29500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-383216}"

if [ "${WIDE:-0}" = 1 ]; then
  MEM_FRACTION="${MEM_FRACTION:-0.90}"
  MAMBA_RATIO="${MAMBA_RATIO:-3.2}"
  MAMBA_CACHE_STRATEGY="${MAMBA_CACHE_STRATEGY:-extra_buffer_lazy}"
fi
MEM_FRACTION="${MEM_FRACTION:-0.88}"
MAMBA_RATIO="${MAMBA_RATIO:-}"
MAMBA_CACHE_STRATEGY="${MAMBA_CACHE_STRATEGY:-}"

mkdir -p "$K3_LOG_DIR/hf" "$K3_LOG_DIR/triton" "$K3_LOG_DIR/inductor"
LOG="$K3_LOG_DIR/k3-rank$RANK.log"

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
BINDS=(-B "$MODEL:$MODEL:ro" -B "$K3_LOG_DIR:$K3_LOG_DIR" -B "$LAUNCHER:$LAUNCHER:ro")

if [ "${K3_PARSER_PATCH:-0}" = 1 ]; then
  _pp="$S/patches"
  _pp_rp=/sgl-workspace/sglang/python/sglang/srt/parser/reasoning_parser.py
  _pp_fmt=/sgl-workspace/sglang/python/sglang/srt/function_call/kimik3_format.py
  for _f in reasoning_parser.py kimik3_format.py; do
    [ -f "$_pp/$_f" ] || { echo "K3_PARSER_PATCH=1 but $_pp/$_f is missing" >&2; exit 1; }
  done
  BINDS+=(-B "$_pp/reasoning_parser.py:$_pp_rp:ro" -B "$_pp/kimik3_format.py:$_pp_fmt:ro")
fi

if [ "$SPEC_MODE" = dspark ]; then
  [ -d "$DRAFT" ] || { echo "SPEC_MODE=dspark needs the draft checkpoint at $DRAFT" >&2; exit 1; }
  BINDS+=(-B "$DRAFT:$DRAFT:ro")
  SPEC=(--speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT")
fi

{
  echo "=== rank $RANK, $(date '+%H:%M:%S') on $(hostname -s)"
  echo "=== head $HEAD_IB:$DIST_PORT, hca $_hca, context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
} >> "$LOG"

exec >> "$LOG" 2>&1
exec singularity exec --nv \
  "${BINDS[@]}" \
  --env NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
  --env GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
  --env NCCL_IB_HCA="$_hca" \
  --env NCCL_DEBUG=WARN \
  --env TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="$TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" \
  --env HF_HOME="$HF_HOME" \
  --env TRITON_CACHE_DIR="$TRITON_CACHE_DIR" \
  --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
  --env PYTHONDONTWRITEBYTECODE="$PYTHONDONTWRITEBYTECODE" \
  "$SIF" \
  python3 "$REPO_ROOT/common/tools/sglang_launch.py" \
    --model-path "$MODEL" \
    --served-model-name kimi-k3 \
    --trust-remote-code \
    --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3 \
    --tp-size 16 \
    --ep-size 16 \
    --nnodes 4 \
    --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IB:$DIST_PORT" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype bf16 \
    --weight-loader-disable-mmap \
    --moe-runner-backend marlin \
    ${MAMBA_RATIO:+--mamba-full-memory-ratio "$MAMBA_RATIO"} \
    ${MAMBA_CACHE_STRATEGY:+--mamba-radix-cache-strategy "$MAMBA_CACHE_STRATEGY"} \
    "${SPEC[@]}"
