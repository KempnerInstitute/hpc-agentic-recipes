#!/usr/bin/env bash
# Bring up DeepSeek-V4-Flash-0731 over SSH on an RTX node you already hold. Secondary path: prefer
# serve.sbatch.
#   bash recipes/DeepSeek-V4-Flash-0731/rtx-8/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
NODE="${1:-${NODE:-${DSV4_FLASH_NODE:-}}}"
[ -n "$NODE" ] || { echo "usage: $(basename "$0") <node>   (or set NODE, or DSV4_FLASH_NODE in common/site.conf)" >&2; exit 2; }
LOG="$LOG_DIR/vllm-deepseek-v4-flash.log"
echo "launching deepseek-v4-flash on $NODE:${API_PORT}"
ssh -o BatchMode=yes "$NODE" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' PERF='${PERF:-}' CUDAGRAPH_MODE='${CUDAGRAPH_MODE:-}' SPEC_TOKENS='${SPEC_TOKENS:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' VENV_DIR='${VENV_DIR:-}' CUDA13_DIR='${CUDA13_DIR:-}' nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: deepseek-v4-flash)"
