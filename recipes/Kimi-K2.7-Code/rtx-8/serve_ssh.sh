#!/usr/bin/env bash
# Bring up Kimi-K2.7-Code over SSH on an RTX node you already hold. Secondary path: prefer
# serve.sbatch. Reserved nodes are removed from the scheduler, which is why this exists.
#   bash recipes/Kimi-K2.7-Code/rtx-8/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${KIMI_RTX_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or KIMI_RTX_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-kimi-rtx.log"
echo "launching kimi-k2.7-code on $NODE:${API_PORT}"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' ENFORCE_EAGER='${ENFORCE_EAGER-1}' ATTN_BACKEND='${ATTN_BACKEND:-}' VENV_DIR='${VENV_DIR:-}' CUDA13_DIR='${CUDA13_DIR:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: kimi-k2.7-code)"
