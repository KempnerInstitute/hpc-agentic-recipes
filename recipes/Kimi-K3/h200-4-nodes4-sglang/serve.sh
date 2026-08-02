#!/usr/bin/env bash
# Start one rank of a 4-node, TP16 Kimi-K3 endpoint inside the SGLang container.
#
#   RANK=<0..3> HEAD_IB=<head node ib0 address> bash serve.sh
#
# One invocation per node, differing only in RANK. serve.sbatch and serve_ssh.sh do that for you.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
source "$REPO_ROOT/common/lib/api_key.sh"

MODEL="${MODEL:-$MODELS_DIR/Kimi-K3}"
DRAFT="${DRAFT:-$MODELS_DIR/Kimi-K3-DSpark}"
RANK="${RANK:?set RANK, 0 through 3}"
HEAD_IB="${HEAD_IB:?set HEAD_IB, the head node ib0 address}"
SPEC_MODE="${SPEC_MODE:-none}"
API_PORT="${API_PORT:-8000}"
DIST_PORT="${DIST_PORT:-29500}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

# WIDE=1 selects the configuration that lifted the concurrency cap from 67 to 156. It is one switch
# rather than three knobs because all three settings are needed together and setting only one does not
# reach that result: the state pool has to grow, the cheaper cache strategy has to cut the per-request
# slot count from 5 to 4, and the static budget has to grow to pay for both. Each is still individually
# overridable. Untested from this recipe, which is why it is not the default.
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

# List only InfiniBand ports that are actually up. mlx5_0 and mlx5_1 on these nodes are dead 40 Gb QDR
# ports while mlx5_2 through mlx5_5 are live 400 Gb NDR, and leaving a down HCA in the list stalls
# initialization. The active set is a property of the node, so it is detected rather than hardcoded.
_hca=""
for _d in /sys/class/infiniband/*; do
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  case "$_s$_l" in *ACTIVE*InfiniBand*) _hca+="${_hca:+,}$(basename "$_d")" ;; esac
done

SPEC=()
if [ "$SPEC_MODE" = dspark ]; then
  # gamma comes from the draft checkpoint's block_size of 7, so the verify window is 8. Letting it
  # auto-infer avoids a mismatch between the flag and the weights. DSpark requires pp_size 1, which is
  # why this is TP16 across four nodes rather than any TP8 by PP2 split.
  SPEC=(--speculative-algorithm DSPARK --speculative-draft-model-path "$DRAFT")
fi

# --trust-remote-code is required even though SGLang registers KimiK3Config and implements
# DSparkDraftModel itself. The blocker is the tokenizer, not the model: tokenizer_config.json maps
# AutoTokenizer to TikTokenTokenizer in tokenization_kimi.py, and transformers' processor loader refuses
# custom code without it. The vocabulary is the local tiktoken.model, so this needs no network.
#
# No --reasoning-parser: K3 always emits reasoning, and a parser moves that text into reasoning_content,
# where a first-token measurement watching content would miss it. Unparsed, every generated token counts
# the same way and time to first token stays meaningful.
#
# --moe-runner-backend marlin is mandatory on Hopper, not a tuning choice. With auto, SGLang picks
# Mxfp4MoEMethod, whose fallback branch calls upcast_from_mxfp to dequantize every expert to bf16, and
# the model then needs roughly 4x its on-disk size: it OOMed with 135 of 139.8 GiB per GPU consumed by
# weights alone. There is an SM90 path that keeps 4 bits, FlashInfer cutlass_fused_moe with
# use_w4_group_scaling, but the flashinfer in this image lacks the interleave_moe helpers it needs, so
# Marlin W4A16 is the only 4-bit-preserving option here. MEM_FRACTION is 0.88: the static budget also
# feeds the KDA state pool, and lowering it to 0.80 dropped the concurrency cap from 67 to 27 while
# leaving decode rate unchanged.
#
# --ep-size 16 matters for memory, not just speed. Under pure TP16 each rank gets 3072/16 = 192 of the
# MoE intermediate dimension, and 192 is not a multiple of the 128 that Marlin needs for a contraction
# dim, so w2 pads to 256. Weights then measured 131.62 GB per GPU against 97.5 expected, 94 percent of
# the card, and the KDA state cache could not be allocated at all: total_rest_memory came out negative
# at -21.18 GB. With expert parallelism each rank holds whole experts, so w2 keeps K=3072 and no padding.
#
# --weight-loader-disable-mmap because the first attempt loaded at about 80 seconds per shard, a 2 hour
# load, with the node 90 percent idle and 7.7 percent in iowait and loader threads parked in D state.
# That is mmap paging 1.56 TB in small random reads over a network filesystem, not MXFP4 unpacking.
#
# No --enable-multimodal: it is opt-in here, and leaving the vision tower out avoids the multimodal
# profiling stall this model family caused on more than one node, with nothing lost for coding.
{
  echo "=== rank $RANK, $(date '+%H:%M:%S') on $(hostname -s)"
  echo "=== head $HEAD_IB:$DIST_PORT, hca $_hca, context $MAX_MODEL_LEN, mem-fraction $MEM_FRACTION"
} >> "$LOG"

exec >> "$LOG" 2>&1
exec singularity exec --nv \
  -B "$MODEL:$MODEL:ro" -B "$DRAFT:$DRAFT:ro" -B "$K3_LOG_DIR:$K3_LOG_DIR" \
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
  python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --served-model-name kimi-k3 \
    --trust-remote-code \
    --tp-size 16 \
    --ep-size 16 \
    --nnodes 4 \
    --node-rank "$RANK" \
    --dist-init-addr "$HEAD_IB:$DIST_PORT" \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --api-key "$VLLM_API_KEY" \
    --context-length "$MAX_MODEL_LEN" \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype bf16 \
    --weight-loader-disable-mmap \
    --moe-runner-backend marlin \
    ${MAMBA_RATIO:+--mamba-full-memory-ratio "$MAMBA_RATIO"} \
    ${MAMBA_CACHE_STRATEGY:+--mamba-radix-cache-strategy "$MAMBA_CACHE_STRATEGY"} \
    "${SPEC[@]}"
