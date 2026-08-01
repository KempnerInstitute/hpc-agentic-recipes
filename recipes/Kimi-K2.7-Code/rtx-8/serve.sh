#!/usr/bin/env bash
# Serve Kimi-K2.7-Code on one RTX PRO 6000 Blackwell node: TP8, eager, native INT4 experts.
# Eager is the default because CUDA graph capture has never been tried for this model, and the
# measured rate below is the eager one. ENFORCE_EAGER= (set but empty) drops --enforce-eager to try
# graph capture, which should be faster but is unverified. The checkpoint is multimodal, so the
# MoonViT encoder is placed with --mm-encoder-tp-mode data rather than tensor sharded.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K2.7-Code}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA=()
# ${VAR-default} rather than ${VAR:-default}: an explicitly empty ENFORCE_EAGER must stay empty and
# turn the flag off, while an unset one keeps the tested default.
[ -n "${ENFORCE_EAGER-1}" ] && EXTRA+=(--enforce-eager)
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)

exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.7-code \
  --tensor-parallel-size "${TP:-8}" \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-kimi_k2}" \
  --reasoning-parser "${REASONING_PARSER:-kimi_k2}" \
  "${EXTRA[@]}"
