#!/usr/bin/env bash
# Bring up Ripple-65B-Preview over SSH on two GPUs of a node you already hold.
#   bash recipes/Ripple-65B-Preview/h200-2/serve_ssh.sh <node> [gpu_list]
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -ge 1 ] || { echo "usage: $(basename "$0") <node> [gpu_list, default 0,1]" >&2; exit 2; }
[ $# -le 2 ] || { echo "too many arguments: this takes a node and an optional GPU list only" >&2; exit 2; }
NODE="$1"; GPUS="${2:-0,1}"
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Ripple-65B-Preview}"
DRAFT="${DRAFT:-$MODELS_DIR/Ripple-65B-Preview-DFlash2}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

# The other GPUs of this node may belong to someone else, so refuse a list that is already busy.
BUSY="$(ssh -o BatchMode=yes "$NODE" "
  for i in \$(echo '$GPUS' | tr ',' ' '); do
    u=\$(nvidia-smi -i \$i --query-compute-apps=used_memory --format=csv,noheader 2>/dev/null | head -1)
    [ -n \"\$u\" ] && echo \"\$i(\$u)\"
  done | tr '\n' ' '")"
if [ -n "${BUSY// /}" ]; then
  echo "GPUs already in use on $NODE: $BUSY" >&2
  echo "pick a free list, or stop that process first" >&2
  exit 1
fi

ssh -o BatchMode=yes "$NODE" "mkdir -p '$RIPPLE_LOG_DIR'; cd '$REPO_ROOT'; \
  CUDA_VISIBLE_DEVICES='$GPUS' API_PORT='$API_PORT' SIF='$SIF' MODELS_DIR='$MODELS_DIR' \
  MODEL='$MODEL' DRAFT='$DRAFT' SPEC_MODE='${SPEC_MODE:-none}' SPEC_TOKENS='${SPEC_TOKENS:-}' SPEC_WINDOW='${SPEC_WINDOW:-}' ATTN='${ATTN:-}' PAGE_SIZE='${PAGE_SIZE:-}' \
  MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' MEM_FRACTION='${MEM_FRACTION:-}' \
  TP='${TP:-}' MAX_RUNNING='${MAX_RUNNING:-}' \
  KEY_NAME='${KEY_NAME:-}' KEY_FILE='${KEY_FILE:-}' RIPPLE_LOG_DIR='$RIPPLE_LOG_DIR' \
  EXTRA_ARGS='${EXTRA_ARGS:-}' \
  nohup bash '$S/serve.sh' > /dev/null 2>&1 < /dev/null & echo '  launched on $NODE gpus $GPUS'"

echo "watch:    ssh $NODE tail -f $RIPPLE_LOG_DIR/ripple-sglang.log"
echo "endpoint: http://$NODE:$API_PORT   (served model: ripple)"
