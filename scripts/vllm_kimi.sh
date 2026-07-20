#!/usr/bin/env bash
# Serve Kimi-K2.7-Code (1T MoE, native INT4 compressed-tensors experts, multimodal MoonViT).
# Select the runtime with ENV_LIB: lib_env.sh (H200, CUDA 12.9) or lib_env_cu130.sh (RTX, sm_120).
# For a multi-node Ray cluster set RAY=1 and pass <head_ip> [ray_port]; single node leaves RAY unset.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/${ENV_LIB:-lib_env.sh}"
MODEL="${MODEL:?set MODEL}"
EXTRA=()
if [ -n "${RAY:-}" ]; then
  export RAY_ADDRESS="${1:?head ip}:${2:-6379}"
  EXTRA+=(--distributed-executor-backend ray --disable-custom-all-reduce)
fi
[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
[ -n "${PP:-}" ] && EXTRA+=(--pipeline-parallel-size "$PP")
[ -n "${ENFORCE_EAGER:-}" ] && EXTRA+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.7-code \
  --tensor-parallel-size "${TP:-8}" \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-kimi_k2}" \
  --reasoning-parser "${REASONING_PARSER:-kimi_k2}" \
  "${EXTRA[@]}"
