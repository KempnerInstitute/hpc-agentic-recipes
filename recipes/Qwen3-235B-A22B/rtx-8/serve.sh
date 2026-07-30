#!/usr/bin/env bash
# Serve Qwen3-235B-A22B on one RTX PRO 6000 Blackwell node: bf16, TP8, CUDA graphs, prefix caching.
# This is a thinking model, so a reasoning parser is always passed. Do not add
# --enable-expert-parallel: the node has no NVLink and the extra all-to-all traffic measured slower
# than plain tensor parallelism. QUANT=fp8 quantizes the weights on load, which measured no faster
# because decode here is limited by cross-GPU communication rather than weight bandwidth.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Qwen3-235B-A22B}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")

exec vllm serve "$MODEL" \
  --served-model-name qwen3-235b \
  --tensor-parallel-size "${TP:-8}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-hermes}" \
  --reasoning-parser "${REASONING_PARSER:-qwen3}" \
  "${EXTRA[@]}"
