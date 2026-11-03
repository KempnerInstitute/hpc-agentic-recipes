#!/usr/bin/env bash
# Serve MiMo-V2.6-Pro-RL on one RTX PRO 6000 node: TP8 inside the node.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-MiMo-V2.6-Pro-RL-rtx-8}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/MiMo-V2.6-Pro-RL}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_UTIL="${GPU_UTIL:-0.92}"

[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
[ -x "$VENV/bin/vllm" ] || { echo "environment not built; run env/build.sh" >&2; exit 1; }

EAGER=()
[ -n "${ENFORCE_EAGER:-}" ] && EAGER=(--enforce-eager)

# TP8 must stay inside one node and pipeline parallelism must not be used: see Known limits.
exec vllm serve "$MODEL" \
  --served-model-name mimo-v2.6-pro \
  --trust-remote-code \
  --tensor-parallel-size "${TP:-8}" \
  --host 0.0.0.0 --port "$API_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config vllm \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-mimo}" \
  --reasoning-parser "${REASONING_PARSER:-mimo}" \
  "${EAGER[@]}" ${EXTRA_ARGS:-}
