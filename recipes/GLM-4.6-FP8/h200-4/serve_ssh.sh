#!/usr/bin/env bash
# Bring up GLM-4.6-FP8 over SSH on a node you already hold. Secondary path: prefer serve.sbatch.
# Reserved nodes are removed from the scheduler, which is why this exists.
#   bash recipes/GLM-4.6-FP8/h200-4/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${GLM46_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or GLM46_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-glm46.log"
echo "launching glm-4.6 on $NODE:${API_PORT}"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' PERF='${PERF:-}' NO_MTP='${NO_MTP:-}' MTP_TOKENS='${MTP_TOKENS:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: glm-4.6)"
