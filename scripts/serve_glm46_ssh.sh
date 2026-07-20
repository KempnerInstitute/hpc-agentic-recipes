#!/usr/bin/env bash
# Bring up GLM-4.6 FP8 on a single H200 node (TP=4) over SSH.
# Configurable inputs (defaults in scripts/config.sh): GLM46_NODE, MODEL, API_PORT, MAX_MODEL_LEN, GPU_UTIL.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$GLM46_NODE"
MODEL="${MODEL:-$GLM46_MODEL}"
mkdir -p "$REPO_DIR/logs"
echo "launching GLM-4.6 on $NODE:$API_PORT  (model: $MODEL)"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-131072}' GPU_UTIL='${GPU_UTIL:-0.90}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' NO_MTP='${NO_MTP:-}' MTP_TOKENS='${MTP_TOKENS:-}' PERF='${PERF:-}' nohup bash '$S/vllm_glm46.sh' > '$REPO_DIR/logs/vllm-glm46.log' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $REPO_DIR/logs/vllm-glm46.log"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: glm-4.6)"
