#!/usr/bin/env bash
# Bring up Kimi-K2.7-Code on a single RTX PRO 6000 node over SSH (8 GPUs, sm_120, TP=8).
# Configurable inputs (defaults in scripts/config.sh): KIMI_NODE, MODEL, API_PORT, TP, MAX_MODEL_LEN, GPU_UTIL.
# LOG_DIR defaults to node-local /tmp: the server writes stderr continuously from every rank, and a
# log on a network filesystem puts a blocking write on the critical path of a live endpoint.
# VENV_DIR and CUDA13_DIR relocate the runtime environment, see scripts/lib_env_cu130.sh.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$KIMI_NODE"
MODEL="${MODEL:-$KIMI_MODEL}"

LOG="$LOG_DIR/vllm-kimi-rtx.log"
echo "launching Kimi-K2.7-Code on RTX $NODE:$API_PORT  (model: $MODEL)"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; mkdir -p '$LOG_DIR'; ENV_LIB=lib_env_cu130.sh VENV_DIR='${VENV_DIR:-}' CUDA13_DIR='${CUDA13_DIR:-}' MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-32768}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-8}' ENFORCE_EAGER='${ENFORCE_EAGER:-1}' ATTN_BACKEND='${ATTN_BACKEND:-}' nohup bash '$S/vllm_kimi.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: kimi-k2.7-code)"
