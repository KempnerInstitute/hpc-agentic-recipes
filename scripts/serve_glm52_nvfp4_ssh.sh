#!/usr/bin/env bash
# Bring up GLM-5.2-NVFP4 on a single RTX PRO 6000 Blackwell node (8 GPUs, sm_120, TP=8) over SSH.
# Configurable inputs (defaults in scripts/config.sh): RTX_NODE, MODEL, API_PORT, TP, MAX_MODEL_LEN, GPU_UTIL.
# Slurm alternative: srun --reservation=<resv> --account=<acct> -p <rtx-partition> -w "$RTX_NODE" \
#   --gres=gpu:8 --cpus-per-task=32 --mem=0 --pty bash ; then: bash scripts/vllm_glm52_nvfp4.sh
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
NODE="$RTX_NODE"
MODEL="${MODEL:-$GLM52_NVFP4_MODEL}"
mkdir -p "$REPO_DIR/logs"
echo "launching GLM-5.2-NVFP4 on $NODE:$API_PORT  (model: $MODEL)"
ssh -o BatchMode=yes "$NODE" "cd '$REPO_DIR'; MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-131072}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-8}' NO_MTP='${NO_MTP:-}' MTP_TOKENS='${MTP_TOKENS:-}' TOOL_PARSER='${TOOL_PARSER:-}' ATTN_BACKEND='${ATTN_BACKEND:-}' nohup bash '$S/vllm_glm52_nvfp4.sh' > '$REPO_DIR/logs/vllm-glm52-nvfp4.log' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $REPO_DIR/logs/vllm-glm52-nvfp4.log"
echo "endpoint: http://$NODE:$API_PORT/v1  (served model: glm-5.2, NVFP4)"
