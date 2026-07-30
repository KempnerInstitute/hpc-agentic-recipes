#!/usr/bin/env bash
# Serve GLM-5.2-NVFP4 on one RTX PRO 6000 Blackwell node: TP8, CUDA graphs, MTP speculative decoding.
# CUDA graphs capture cleanly on sm_120 with CUDA 13, so there is no eager fallback here, and one node
# means no pipeline parallelism, which is what keeps speculative decoding available. vLLM auto-selects
# the FlashInfer sparse-MLA sm_120 backend for GLM-5.2's sparse attention; ATTN_BACKEND overrides that
# choice, and NO_MTP=1 disables speculative decoding.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/GLM-5.2-NVFP4}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
EXTRA=()
[ -z "${NO_MTP:-}" ] && EXTRA+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": ${MTP_TOKENS:-3}}")

exec vllm serve "$MODEL" \
  --served-model-name glm-5.2 \
  --tensor-parallel-size "${TP:-8}" \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_ds_mla \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-131072}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-glm45}" \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  "${EXTRA[@]}"
