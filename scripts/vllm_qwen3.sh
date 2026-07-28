#!/usr/bin/env bash
# Serve a Qwen3 MoE checkpoint. Two shapes are supported through TP and PP:
#   Qwen3-235B-A22B       : TP8 on one RTX node (all 8 GPUs)
#   Qwen3-Coder-480B-FP8  : TP4 x PP2 on one RTX node
# The FP8 Coder checkpoint cannot use TP8: moe_intermediate_size is 2560 and its FP8 quantization
# block is 128, so 2560/8 = 320 is not divisible by 128 and vLLM refuses to start. TP4 is required,
# which on an 8-GPU node means adding PP2.
# Native context for the 235B is 40960; the Coder supports 262144.
# Do not enable expert parallelism on the RTX node: it has no NVLink, so the extra all-to-all traffic
# measured slower than plain tensor parallelism. FP8 weights measured no faster on the 235B either,
# because decode is limited by cross-GPU communication rather than weight bandwidth.
set -euo pipefail
source "$(dirname "$0")/${ENV_LIB:-lib_env_cu130.sh}"
MODEL="${MODEL:?set MODEL (launch via serve_qwen3_ssh.sh or slurm_qwen3*.sbatch)}"
EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")
[ -n "${PP:-}" ] && EXTRA+=(--pipeline-parallel-size "$PP" --distributed-executor-backend "${EXECUTOR:-mp}")
# Qwen3-235B is a thinking model and needs a reasoning parser. Qwen3-Coder-Instruct is not, so pass
# REASONING_PARSER= to omit the flag rather than mis-parsing plain output as reasoning.
_RP="${REASONING_PARSER-qwen3}"
[ -n "$_RP" ] && EXTRA+=(--reasoning-parser "$_RP")
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
  "${EXTRA[@]}"
