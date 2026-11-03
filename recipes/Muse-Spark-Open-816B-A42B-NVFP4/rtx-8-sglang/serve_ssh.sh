#!/usr/bin/env bash
# Bring up Muse-Spark-Open-816B-A42B-NVFP4 over SSH on one RTX node you already hold.
#   bash recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang/serve_ssh.sh <node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 1 ] || { echo "usage: $(basename "$0") <node>" >&2; exit 2; }
NODE="$1"
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4}"
DRAFT="${DRAFT:-$MODELS_DIR/Muse-Spark-Open-DFlash2}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

ssh -o BatchMode=yes "$NODE" "mkdir -p '$MUSE_LOG_DIR'; cd '$REPO_ROOT'; \
  API_PORT='$API_PORT' SIF='$SIF' MODELS_DIR='$MODELS_DIR' \
  MODEL='$MODEL' DRAFT='$DRAFT' NO_SPEC='${NO_SPEC:-0}' \
  MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' MEM_FRACTION='${MEM_FRACTION:-}' \
  TP='${TP:-}' EP='${EP:-}' MOE_RUNNER='${MOE_RUNNER:-}' ATTN='${ATTN:-}' \
  KV_DTYPE='${KV_DTYPE:-}' PAGE_SIZE='${PAGE_SIZE:-}' CHUNKED_PREFILL='${CHUNKED_PREFILL:-}' \
  KEY_NAME='${KEY_NAME:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' \
  nohup bash '$S/serve.sh' > /dev/null 2>&1 < /dev/null & echo '  launched on $NODE'"

echo "watch:    ssh $NODE tail -f $MUSE_LOG_DIR/muse-nvfp4-sglang.log"
echo "endpoint: http://$NODE:$API_PORT   (served model: muse-spark)"
