#!/usr/bin/env bash
# Bring up a Gemma 4 checkpoint on one GPU over SSH (TP1).
# Configurable inputs (defaults in scripts/config.sh): GEMMA4_NODE, MODEL, API_PORT, MAX_MODEL_LEN.
# Set GPU to pin a device on a shared node, for example GPU=1.
# Defaults serve the 26B-A4B MoE in bf16. For the dense 31B, which wants FP8:
#   MODEL="$GEMMA31_MODEL" SERVED_NAME=gemma-4-31b QUANT=fp8 bash scripts/serve_gemma4_ssh.sh
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$GEMMA4_NODE"
MODEL="${MODEL:-$GEMMA4_MODEL}"
SERVED_NAME="${SERVED_NAME:-gemma-4-26b}"
mkdir -p "$REPO_DIR/logs"
LOG="$REPO_DIR/logs/vllm-$SERVED_NAME.log"
echo "launching $SERVED_NAME on $NODE:$API_PORT  (model: $MODEL, GPU ${GPU:-all}, quant ${QUANT:-bf16})"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} MODEL='$MODEL' SERVED_NAME='$SERVED_NAME' QUANT='${QUANT:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-32768}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-1}' SPEC_DRAFT='${SPEC_DRAFT:-}' SPEC_TOKENS='${SPEC_TOKENS:-}' KV_FP8='${KV_FP8:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/vllm_gemma4.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: $SERVED_NAME)"
