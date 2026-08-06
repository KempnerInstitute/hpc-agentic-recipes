#!/usr/bin/env bash
# Bring up gemma-4-26B-A4B-it over SSH on a node you already hold. Secondary path: prefer serve.sbatch.
#   bash recipes/gemma-4-26B-A4B-it/h200-1/serve_ssh.sh <node>
# This recipe uses one GPU, so set GPU=<n> to pin a device on a node whose other GPUs are busy.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${GEMMA26_H200_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or GEMMA26_H200_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-gemma-4-26b-$API_PORT.log"
echo "launching gemma-4-26b on $NODE:${API_PORT}   (GPU ${GPU:-all})"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; ${GPU:+CUDA_VISIBLE_DEVICES=$GPU} MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' QUANT='${QUANT:-}' KV_FP8='${KV_FP8:-}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' SPEC_DRAFT='${SPEC_DRAFT:-}' SPEC_TOKENS='${SPEC_TOKENS:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: gemma-4-26b)"
