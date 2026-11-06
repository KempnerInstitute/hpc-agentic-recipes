#!/usr/bin/env bash
# Serve MiniMax-M2.7 on one H200 node: TP4, FP8.
# Upstream's default is exactly TP4 on four H200 GPUs, which is one node here. Pure TP8 is not
# supported for this architecture; TP8 needs --enable-expert-parallel so the expert tensors shard
# evenly, so do not raise TP without reading Known limits.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/MiniMax-M2.7}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

EXTRA=()
[ -n "${EXTRA_ARGS:-}" ] && EXTRA+=($EXTRA_ARGS)
if [ -n "${PERF:-}" ]; then
  EXTRA+=(--compilation-config "{\"cudagraph_mode\": \"${CUDAGRAPH_MODE:-NONE}\"}")
else
  EXTRA+=(--enforce-eager)
fi
# No MTP draft is published for this architecture. vLLM registers Eagle3MiniMaxM2ForCausalLM, so a
# draft could be used, but none is downloaded or measured here, so speculation is off.
[ -n "${SPEC_CONFIG:-}" ] && EXTRA+=(--speculative-config "$SPEC_CONFIG")

export VLLM_USE_RUST_FRONTEND="${VLLM_USE_RUST_FRONTEND:-1}"

exec vllm serve "$MODEL" \
  --served-model-name minimax-m2.7 \
  --tensor-parallel-size "${TP:-4}" \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-196608}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-minimax_m2}" \
  --reasoning-parser "${REASONING_PARSER:-minimax_m2}" \
  "${EXTRA[@]}"
