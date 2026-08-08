#!/usr/bin/env bash
# Serve Kimi-K2.7-Code on one RTX PRO 6000 Blackwell node: TP8, eager.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K2.7-Code}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA=()
[ -n "${ENFORCE_EAGER-1}" ] && EXTRA+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.7-code \
  --tensor-parallel-size "${TP:-8}" \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-kimi_k2}" \
  --reasoning-parser "${REASONING_PARSER:-kimi_k2}" \
  "${EXTRA[@]}"
