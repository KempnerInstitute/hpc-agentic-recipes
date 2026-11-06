#!/usr/bin/env bash
# VLLM_API_KEY is deliberately not forwarded: it would appear in the remote process command
# line. KEY_NAME and KEY_FILE are forwarded instead and the key is resolved on the node.
# Bring up MiniMax-M2.7 over SSH on a node you already hold.
#   bash recipes/MiniMax-M2.7/h200-4/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${MINIMAX_M27_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or MINIMAX_M27_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-minimax-m2.7.log"
echo "launching minimax-m2.7 on $NODE:${API_PORT}"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' MODELS_DIR='${MODELS_DIR:-}' KEY_NAME='${KEY_NAME:-}' KEY_FILE='${KEY_FILE:-}' API_PORT='$API_PORT' CUDAGRAPH_MODE='${CUDAGRAPH_MODE:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' PERF='${PERF:-}' SPEC_CONFIG='${SPEC_CONFIG:-}' VLLM_USE_RUST_FRONTEND='${VLLM_USE_RUST_FRONTEND:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: minimax-m2.7)"
