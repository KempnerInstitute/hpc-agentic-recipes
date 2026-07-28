#!/usr/bin/env bash
# Bring up Gemma-4-26B-A4B-it on one GPU of an RTX PRO 6000 node over SSH (TP1).
# Configurable inputs (defaults in scripts/config.sh): GEMMA4_NODE, MODEL, API_PORT, MAX_MODEL_LEN.
# Set GPU to pin a specific device when the node is shared, for example GPU=1.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$GEMMA4_NODE"
MODEL="${MODEL:-$GEMMA4_MODEL}"
mkdir -p "$REPO_DIR/logs"
echo "launching Gemma-4-26B on $NODE:$API_PORT  (model: $MODEL, GPU ${GPU:-all})"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-32768}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-1}' SPEC_DRAFT='${SPEC_DRAFT:-}' KV_FP8='${KV_FP8:-}' nohup bash '$S/vllm_gemma4.sh' > '$REPO_DIR/logs/vllm-gemma4.log' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $REPO_DIR/logs/vllm-gemma4.log"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: gemma-4-26b)"
