#!/usr/bin/env bash
# Serve Qwen3-235B-A22B across all 8 GPUs of one RTX PRO 6000 node (TP8).
# Native context is 40960; longer needs YaRN scaling.
# Do not enable expert parallelism here: the node has no NVLink, so the extra all-to-all traffic
# measured slower than plain tensor parallelism. FP8 weights measured no faster either, because
# decode is limited by cross-GPU communication rather than weight bandwidth.
set -euo pipefail
source "$(dirname "$0")/${ENV_LIB:-lib_env_cu130.sh}"
MODEL="${MODEL:?set MODEL (launch via serve_qwen3_ssh.sh or slurm_qwen3.sbatch)}"
EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
exec vllm serve "$MODEL" \
  --served-model-name "${SERVED_NAME:-qwen3-235b}" \
  --tensor-parallel-size "${TP:-8}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-hermes}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${EXTRA[@]}"
