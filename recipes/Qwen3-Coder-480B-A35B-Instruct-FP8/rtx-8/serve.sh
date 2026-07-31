#!/usr/bin/env bash
# Serve Qwen3-Coder-480B-A35B-Instruct-FP8 on one RTX PRO 6000 Blackwell node: TP4 x PP2, FP8, CUDA
# graphs, prefix caching.
# TP4 is mandatory for this FP8 checkpoint: moe_intermediate_size is 2560 and the FP8 quantization
# block is 128, so TP8 leaves 2560/8 = 320 per shard, which is not a multiple of 128, and vLLM refuses
# to start. On an 8-GPU node that means PP2 to use the other four GPUs.
# This is not a thinking model, so no reasoning parser is passed.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Qwen3-Coder-480B-A35B-Instruct-FP8}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=(--pipeline-parallel-size "${PP:-2}" --distributed-executor-backend "${EXECUTOR:-mp}")
# ${VAR-default} rather than ${VAR:-default}: the default here is no reasoning parser, and an
# explicitly empty REASONING_PARSER must stay empty rather than fall back to one. Parsing plain
# output as reasoning would move the answer into a field most clients never display.
_RP="${REASONING_PARSER-}"
[ -n "$_RP" ] && EXTRA+=(--reasoning-parser "$_RP")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name qwen3-coder-480b \
  --tensor-parallel-size "${TP:-4}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-qwen3_coder}" \
  "${EXTRA[@]}"
