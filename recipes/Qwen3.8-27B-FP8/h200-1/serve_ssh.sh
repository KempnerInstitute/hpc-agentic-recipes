#!/usr/bin/env bash
# Bring up Qwen3.8-27B-FP8 over SSH on a node you already hold. Secondary path: prefer serve.sbatch.
#   bash recipes/Qwen3.8-27B-FP8/h200-1/serve_ssh.sh <node>
# This recipe uses one GPU, so set GPU=<n> to pin a device on a node whose other GPUs are busy.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${QWEN38_27B_FP8_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or QWEN38_27B_FP8_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-qwen3.8-27b-fp8-$API_PORT.log"
echo "launching qwen3.8-27b-fp8 on $NODE:${API_PORT}   (GPU ${GPU:-all})"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' KV_DTYPE='${KV_DTYPE:-}' KV_FP8='${KV_FP8:-}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' MTP_TOKENS='${MTP_TOKENS:-}' SPEC_METHOD='${SPEC_METHOD:-}' CUDAGRAPH_MODE='${CUDAGRAPH_MODE:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: qwen3.8-27b-fp8)"
