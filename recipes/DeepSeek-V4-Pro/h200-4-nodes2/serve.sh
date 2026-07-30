#!/usr/bin/env bash
# Serve DeepSeek-V4-Pro across two H200 nodes: TP4 inside each node, PP2 between them, Ray.
#   bash recipes/DeepSeek-V4-Pro/h200-4-nodes2/serve.sh <head_ip> [ray_port]
# Run this on the Ray head after both nodes have joined the cluster; serve.sbatch does that for you.
#
# This configuration is expected to be problematic and is kept to record the result. The checkpoint's
# routed experts are FP4 and Hopper has no FP4 hardware, so the expert layers either fall back to a
# slower path or fail outright. The recommended variant is recipes/DeepSeek-V4-Pro/rtx-8-nodes2.
# TP4 satisfies the FP8 block constraint: moe_intermediate_size 3072 divided by 4 is 768, a multiple
# of the 128 quantization block.
# No speculative decoding: the checkpoint ships an MTP head, but pipeline parallelism is required to
# span nodes and vLLM rejects a speculative config when pipeline parallelism is active.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/DeepSeek-V4-Pro}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }
HEAD_IP="${1:-${RAY_HEAD_IP:-}}"
[ -n "$HEAD_IP" ] || { echo "usage: $(basename "$0") <head_ip> [ray_port]   (or set RAY_HEAD_IP)" >&2; exit 2; }
export RAY_ADDRESS="$HEAD_IP:${2:-${RAY_PORT:-6379}}"

EXTRA=()
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

# verified 2026-07-29: without --kv-cache-dtype fp8_ds_mla the engine aborts during init with
# "DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto". This model uses MLA with
# DeepSeek sparse attention, and that layout requires an fp8 KV cache rather than the default.
# GLM-5.2-NVFP4 needs the same flag for the same reason.
exec vllm serve "$MODEL" \
  --served-model-name deepseek-v4-pro \
  --kv-cache-dtype fp8_ds_mla \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-2}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-deepseek_v4}" \
  --reasoning-parser "${REASONING_PARSER:-deepseek_v4}" \
  "${EXTRA[@]}"
