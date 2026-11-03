#!/usr/bin/env bash
# Serve Muse-Glimmer-30B on one H200 GPU.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
KEY_NAME="${KEY_NAME:-Muse-Glimmer-30B-h200-1}"
source "$REPO_ROOT/common/lib/api_key.sh"
MODEL="${MODEL:-$MODELS_DIR/Muse-Glimmer-30B}"
DRAFT="${DRAFT:-$MODELS_DIR/Muse-Glimmer-30B-assistant}"
API_PORT="${API_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
GPU_UTIL="${GPU_UTIL:-0.90}"

[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }
[ -x "$VENV/bin/vllm" ] || { echo "environment not built; run env/build.sh" >&2; exit 1; }

SPEC=()
MAXSEQ=("--max-num-seqs" "${MAX_NUM_SEQS:-256}")
if [ "${SPEC_MODE:-none}" = dflash ]; then
  [ -d "$DRAFT" ] || { echo "SPEC_MODE=dflash needs the assistant checkpoint at $DRAFT" >&2; exit 1; }
  # The head has block_size 16 and slot 0 re-presents the last accepted token, so 15 is what remains.
  SPEC=(--speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":15}")
  MAXSEQ=("--max-num-seqs" "${MAX_NUM_SEQS:-32}")
fi

EAGER=()
[ -n "${ENFORCE_EAGER:-}" ] && EAGER=(--enforce-eager)

exec vllm serve "$MODEL" \
  --served-model-name muse-glimmer \
  --tensor-parallel-size "${TP:-1}" \
  --host 0.0.0.0 --port "$API_PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --generation-config auto \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-muse_glimmer}" \
  --reasoning-parser "${REASONING_PARSER:-muse_glimmer}" \
  "${MAXSEQ[@]}" "${SPEC[@]}" "${EAGER[@]}" ${EXTRA_ARGS:-}
