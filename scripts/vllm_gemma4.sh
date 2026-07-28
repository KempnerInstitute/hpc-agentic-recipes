#!/usr/bin/env bash
# Serve a Gemma 4 checkpoint on a single GPU (TP1). Works for both the 26B-A4B MoE and the 31B dense
# model; pick one with MODEL, SERVED_NAME, and QUANT.
# QUANT=fp8 halves the weights. It is the right default for the dense 31B, which is memory bandwidth
# bound, and pointless for the 26B MoE, which activates only 4B parameters and is overhead bound.
# Set SPEC_DRAFT to a Gemma 4 assistant checkpoint for MTP speculative decoding; that needs a vLLM
# build newer than 0.26.0 (see README).
set -euo pipefail
source "$(dirname "$0")/${ENV_LIB:-lib_env_cu130.sh}"
MODEL="${MODEL:?set MODEL (launch via serve_gemma4_ssh.sh or slurm_gemma4*.sbatch)}"
EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")
[ -n "${SPEC_DRAFT:-}" ] && EXTRA+=(--speculative-config "{\"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": ${SPEC_TOKENS:-3}}")
[ -n "${KV_FP8:-}" ] && EXTRA+=(--kv-cache-dtype fp8)
[ -n "${ENFORCE_EAGER:-}" ] && EXTRA+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-gemma-4-26b}" \
  --tensor-parallel-size "${TP:-1}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-gemma4}" \
  --reasoning-parser "${REASONING_PARSER:-gemma4}" \
  "${EXTRA[@]}"
