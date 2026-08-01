#!/usr/bin/env bash
# Serve gemma-4-26B-A4B-it on one H100 GPU: TP1, bf16, prefix caching on.
# bf16 is deliberate. This checkpoint activates 4B of its 26B parameters, so it is host overhead bound
# rather than memory bandwidth bound, and FP8 weights measured no change at all. Set QUANT=fp8 only to
# free VRAM, never for speed.
# SPEC_DRAFT wires the gemma-4-26B-A4B-it-assistant drafter. It is broken on vLLM 0.25.1 and 0.26.0,
# so leave it unset; the plumbing stays so the first engine that fixes it needs no code change.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/gemma-4-26B-A4B-it}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=()
[ -n "${QUANT:-}" ] && EXTRA+=(--quantization "$QUANT")
[ -n "${KV_FP8:-}" ] && EXTRA+=(--kv-cache-dtype fp8)
[ -n "${ENFORCE_EAGER:-}" ] && EXTRA+=(--enforce-eager)
[ -n "${SPEC_DRAFT:-}" ] && EXTRA+=(--speculative-config "{\"model\": \"$SPEC_DRAFT\", \"num_speculative_tokens\": ${SPEC_TOKENS:-3}}")
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
