#!/usr/bin/env bash
# Serve gemma-4-31B-it on one RTX PRO 6000 Blackwell GPU: TP1, FP8 weights, prefix caching on.
# FP8 is the default because this checkpoint is dense and memory bandwidth bound, so halving the bytes
# read per token buys most of a proportional speedup. FP8 measured 74 percent faster than bf16 on this GPU, and it frees enough VRAM for a 503K token KV cache instead of 190K.
# Set QUANT= (empty) for bf16 fidelity at a large cost in rate.
# SPEC_DRAFT wires the gemma-4-31B-it-assistant drafter. It is broken on vLLM 0.25.1 and 0.26.0, so
# leave it unset; the plumbing stays so the first engine that fixes it needs no code change.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/gemma-4-31B-it}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
# Set but empty means bf16 on purpose, so this uses ${QUANT-fp8} rather than ${QUANT:-fp8}.
QUANT="${QUANT-fp8}"

EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")
[ -n "${KV_FP8:-}" ] && EXTRA+=(--kv-cache-dtype fp8)
[ -n "${ENFORCE_EAGER:-}" ] && EXTRA+=(--enforce-eager)
[ -n "${SPEC_DRAFT:-}" ] && EXTRA+=(--speculative-config "{\"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": ${SPEC_TOKENS:-3}}")
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name gemma-4-31b \
  --tensor-parallel-size "${TP:-1}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-gemma4}" \
  --reasoning-parser "${REASONING_PARSER:-gemma4}" \
  "${EXTRA[@]}"
