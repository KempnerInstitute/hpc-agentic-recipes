#!/usr/bin/env bash
# Bring up GLM-5.2-NVFP4 over SSH on a node you already hold. Secondary path: prefer serve.sbatch.
#   bash recipes/GLM-5.2-NVFP4/rtx-8/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${GLM52_NVFP4_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or GLM52_NVFP4_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-glm52-nvfp4.log"
echo "launching glm-5.2 on $NODE:${API_PORT}"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' EXTRA_ARGS='${EXTRA_ARGS:-}' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' NO_MTP='${NO_MTP:-}' MTP_TOKENS='${MTP_TOKENS:-}' ATTN_BACKEND='${ATTN_BACKEND:-}' VENV_DIR='${VENV_DIR:-}' CUDA13_DIR='${CUDA13_DIR:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: glm-5.2)"
