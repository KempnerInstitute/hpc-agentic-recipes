#!/usr/bin/env bash
# Bring up MiMo-V2.6-Pro-RL over SSH on one RTX node you already hold.
#   bash recipes/MiMo-V2.6-Pro-RL/rtx-8/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 1 ] || { echo "usage: $(basename "$0") <node>" >&2; exit 2; }
NODE="$1"
API_PORT="${API_PORT:-8000}"
LOG="$MIMO_LOG_DIR/vllm-mimo.log"

ssh -o BatchMode=yes "$NODE" "mkdir -p '$MIMO_LOG_DIR'; cd '$REPO_ROOT'; \
  API_PORT='$API_PORT' MODELS_DIR='$MODELS_DIR' MODEL='${MODEL:-}' \
  MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' \
  ENFORCE_EAGER='${ENFORCE_EAGER:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' \
  KEY_NAME='${KEY_NAME:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' \
  nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo '  launched on $NODE'"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: mimo-v2.6-pro)"
