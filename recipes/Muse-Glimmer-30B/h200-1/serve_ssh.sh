#!/usr/bin/env bash
# Bring up Muse-Glimmer-30B over SSH on one GPU of a node you already hold.
#   bash recipes/Muse-Glimmer-30B/h200-1/serve_ssh.sh <node> [gpu_index]
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <node> [gpu_index]" >&2; exit 2; }
NODE="$1"; GPU="${2:-0}"
API_PORT="${API_PORT:-8000}"
LOG="${GLIMMER_LOG:-/tmp/$USER/glimmer/vllm-glimmer.log}"

ssh -o BatchMode=yes "$NODE" "mkdir -p '$(dirname "$LOG")'; cd '$REPO_ROOT'; \
  CUDA_VISIBLE_DEVICES='$GPU' API_PORT='$API_PORT' MODELS_DIR='$MODELS_DIR' \
  MODEL='${MODEL:-}' DRAFT='${DRAFT:-}' SPEC_MODE='${SPEC_MODE:-none}' \
  MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' \
  MAX_NUM_SEQS='${MAX_NUM_SEQS:-}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' \
  TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' \
  KEY_NAME='${KEY_NAME:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' \
  nohup bash '$S/serve.sh' > '$LOG' 2>&1 < /dev/null & echo '  launched on $NODE gpu $GPU'"
echo "watch:    ssh $NODE tail -f $LOG"
echo "endpoint: http://$NODE:$API_PORT/v1   (served model: muse-glimmer)"
