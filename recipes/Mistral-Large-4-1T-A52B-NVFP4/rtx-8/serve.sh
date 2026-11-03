#!/usr/bin/env bash
# Serve Mistral-Large-4-1T-A52B-NVFP4 on one RTX PRO 6000 node under vLLM, in a container. TP8.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Mistral-Large-4-1T-A52B-NVFP4-rtx-8}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-98304}"
GPU_UTIL="${GPU_UTIL:-0.92}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
mkdir -p "$ML4N_LOG_DIR/hf"

SPEC=()
[ "${NO_SPEC:-0}" = 1 ] || SPEC=(--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_TOKENS:-3}}")

exec singularity exec --nv \
  -B "$MODEL:$MODEL:ro" -B "$ML4N_LOG_DIR:$ML4N_LOG_DIR" \
  --env VLLM_API_KEY="$VLLM_API_KEY" \
  --env HF_HUB_OFFLINE="$HF_HUB_OFFLINE" --env HF_HOME="$HF_HOME" \
  --env VLLM_ENGINE_READY_TIMEOUT_S="$VLLM_ENGINE_READY_TIMEOUT_S" \
  "$SIF" \
  vllm serve "$MODEL" \
    --served-model-name mistral-large-4 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral \
    --tensor-parallel-size "${TP:-8}" \
    --distributed-executor-backend mp \
    --host 0.0.0.0 --port "$API_PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS:-8192}" \
    --max-num-seqs "${MAX_NUM_SEQS:-128}" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser "${TOOL_PARSER:-mistral}" \
    --reasoning-parser "${REASONING_PARSER:-mistral}" \
    "${SPEC[@]}" ${EXTRA_ARGS:-}
