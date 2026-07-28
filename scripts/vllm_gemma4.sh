#!/usr/bin/env bash
# Serve Gemma-4-26B-A4B-it on a single GPU (TP1). Multimodal MoE, 26B total with 4B active.
# Decode is bound by per-step overhead rather than memory bandwidth on this model, so FP8 weights
# and an FP8 KV cache measured no faster than bf16 and are left off. Set SPEC_DRAFT to a Gemma 4
# assistant checkpoint to enable MTP speculative decoding, which needs a vLLM build that shares the
# target embeddings with the drafter (not in 0.25.1 or 0.26.0).
set -euo pipefail
source "$(dirname "$0")/${ENV_LIB:-lib_env_cu130.sh}"
MODEL="${MODEL:?set MODEL (launch via serve_gemma4_ssh.sh or slurm_gemma4.sbatch)}"
EXTRA=()
[ -n "${SPEC_DRAFT:-}" ] && EXTRA+=(--speculative-config "{\"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": ${SPEC_TOKENS:-3}}")
[ -n "${KV_FP8:-}" ] && EXTRA+=(--kv-cache-dtype fp8)
[ -n "${ENFORCE_EAGER:-}" ] && EXTRA+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
exec vllm serve "$MODEL" \
  --served-model-name gemma-4-26b \
  --tensor-parallel-size "${TP:-1}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-gemma4}" \
  --reasoning-parser "${REASONING_PARSER:-gemma4}" \
  "${EXTRA[@]}"
