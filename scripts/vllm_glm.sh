#!/usr/bin/env bash
# Serve GLM-5.2-FP8 on the Ray cluster (TP=4, PP=2). Usage: vllm_glm.sh <head_ip> [ray_port]
# Default: eager (stable). torch.compile / CUDA graphs / DeepGEMM all hit illegal-memory-access
# on GLM-5.2 DSA in vLLM 0.25.1; set PERF=1 to retry the compile path after a vLLM upgrade.
# No MTP: vLLM rejects speculative decoding with pipeline parallelism, which GLM-5.2 needs here.
set -euo pipefail
source "$(dirname "$0")/lib_env.sh"
HEAD_IP="${1:?head ip}"
RAY_PORT="${2:-6379}"
export RAY_ADDRESS="$HEAD_IP:$RAY_PORT"
MODEL="${MODEL:?set MODEL (launch via serve_glm_ssh.sh or slurm_glm52_fp8.sbatch)}"
EXTRA=()
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
exec vllm serve "$MODEL" \
  --served-model-name glm-5.2 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm45}" \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  "${EXTRA[@]}"
