#!/usr/bin/env bash
# Serve GLM-5.2-NVFP4 (full ModelOpt NVFP4) on one RTX PRO 6000 Blackwell node (8x, sm_120, CUDA 13).
# vLLM auto-selects the FlashInfer sparse-MLA sm_120 backend for GLM-5.2's DeepSeek Sparse Attention;
# NVFP4 is near-FP8 quality and MTP speculative decoding works (single node, no pipeline parallelism).
# Native Anthropic API, so Claude Code connects directly. Set ATTN_BACKEND to override backend selection.
set -euo pipefail
source "$(dirname "$0")/lib_env_cu130.sh"
MODEL="${MODEL:?set MODEL (launch via serve_glm52_nvfp4_ssh.sh or slurm_glm52_nvfp4.sbatch)}"
[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA=()
if [ -z "${NO_MTP:-}" ]; then
  EXTRA+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS:-3}}")
fi
exec vllm serve "$MODEL" \
  --served-model-name glm-5.2 \
  --tensor-parallel-size "${TP:-8}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm45}" \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  "${EXTRA[@]}"
