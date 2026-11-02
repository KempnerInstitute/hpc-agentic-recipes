#!/usr/bin/env bash
# Serve GLM-5.3-NVFP4 on one RTX PRO 6000 Blackwell node: TP8, CUDA graphs, MTP speculative decoding.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/GLM-5.3-NVFP4}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA=()
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
[ -z "${NO_MTP:-}" ] && EXTRA+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS:-3}}")

exec vllm serve "$MODEL" \
  --served-model-name glm-5.3 \
  --tensor-parallel-size "${TP:-8}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype "${KV_DTYPE:-fp8_e4m3}" \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-217344}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm47}" \
  --reasoning-parser "${REASONING_PARSER:-glm47}" \
  "${EXTRA[@]}"
