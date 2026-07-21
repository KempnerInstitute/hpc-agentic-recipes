#!/usr/bin/env bash
# Serve GLM-4.6-FP8 single node (TP=4), eager + MTP speculative decoding.
# CUDA graphs / torch.compile crash on this vLLM 0.25.1 / H200 (illegal memory access in capture),
# so the default is eager; MTP (the model's multi-token-prediction head) provides the decode speedup.
# PERF=1 retries the compile path (after a vLLM fix); NO_MTP=1 disables speculative decoding.
set -euo pipefail
source "$(dirname "$0")/lib_env.sh"
MODEL="${MODEL:?set MODEL (launch via serve_glm46_ssh.sh or slurm_glm46.sbatch)}"
EXTRA=()
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
if [ -z "${NO_MTP:-}" ]; then
  EXTRA+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS:-1}}")
fi
exec vllm serve "$MODEL" \
  --served-model-name glm-4.6 \
  --tensor-parallel-size 4 \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm45}" \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  "${EXTRA[@]}"
