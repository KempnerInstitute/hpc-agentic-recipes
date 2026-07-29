#!/usr/bin/env bash
# Bring up Qwen3-235B-A22B on all 8 GPUs of one RTX PRO 6000 node over SSH (TP8).
# Configurable inputs (defaults in scripts/config.sh): QWEN3_NODE, MODEL, API_PORT, MAX_MODEL_LEN.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$QWEN3_NODE"
MODEL="${MODEL:-$QWEN3_MODEL}"
echo "launching Qwen3-235B on $NODE:$API_PORT  (model: $MODEL, TP ${TP:-8})"
SERVED_NAME="${SERVED_NAME:-qwen3-235b}"

LOG="$LOG_DIR/vllm-$SERVED_NAME.log"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; mkdir -p '$LOG_DIR'; VENV_DIR='${VENV_DIR:-}' CUDA13_DIR='${CUDA13_DIR:-}' MODEL='$MODEL' SERVED_NAME='$SERVED_NAME' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-40960}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-8}' PP='${PP:-}' QUANT='${QUANT:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER-qwen3}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/vllm_qwen3.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: $SERVED_NAME)"
