#!/usr/bin/env bash
# Bring up Mistral-Large-4-1T-A52B-NVFP4 over SSH on one RTX node you already hold.
#   bash recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 1 ] || { echo "usage: $(basename "$0") <node>" >&2; exit 2; }
NODE="$1"
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4}"
LOG="$ML4N_LOG_DIR/vllm-mistral-nvfp4.log"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

ssh -o BatchMode=yes "$NODE" "mkdir -p '$ML4N_LOG_DIR'; cd '$REPO_ROOT'; \
  API_PORT='$API_PORT' SIF='$SIF' MODELS_DIR='$MODELS_DIR' MODEL='$MODEL' \
  MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' \
  MAX_BATCHED_TOKENS='${MAX_BATCHED_TOKENS:-}' MAX_NUM_SEQS='${MAX_NUM_SEQS:-}' \
  NO_SPEC='${NO_SPEC:-0}' SPEC_TOKENS='${SPEC_TOKENS:-}' \
  TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' \
  KEY_NAME='${KEY_NAME:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' \
  nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo '  launched on $NODE'"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: mistral-large-4)"
