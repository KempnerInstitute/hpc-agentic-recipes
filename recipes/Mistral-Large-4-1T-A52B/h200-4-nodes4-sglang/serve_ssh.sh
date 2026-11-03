#!/usr/bin/env bash
# Bring up Mistral-Large-4-1T-A52B FP8 over SSH on four H200 nodes you already hold.
#   bash recipes/Mistral-Large-4-1T-A52B/h200-4-nodes4-sglang/serve_ssh.sh <n0> <n1> <n2> <n3>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 4 ] || { echo "usage: $(basename "$0") <node0> <node1> <node2> <node3>" >&2; exit 2; }
NODES=("$@")
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Mistral-Large-4-1T-A52B}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

HEAD_IB="$(ssh -o BatchMode=yes "${NODES[0]}" "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")"
[ -n "$HEAD_IB" ] || { echo "could not read ib0 on ${NODES[0]}" >&2; exit 1; }
echo "head ${NODES[0]} at $HEAD_IB, serving mistral-large-4 on ${NODES[0]}:$API_PORT"

for i in 0 1 2 3; do
  ssh -o BatchMode=yes "${NODES[$i]}" "mkdir -p '$ML4_LOG_DIR'; cd '$REPO_ROOT'; \
    RANK=$i HEAD_IB='$HEAD_IB' API_PORT='$API_PORT' DIST_PORT='$DIST_PORT' SIF='$SIF' \
    MODELS_DIR='$MODELS_DIR' MODEL='$MODEL' \
    MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' MEM_FRACTION='${MEM_FRACTION:-}' \
    TP='${TP:-}' NNODES='${NNODES:-}' MAX_RUNNING='${MAX_RUNNING:-}' NO_SPEC='${NO_SPEC:-0}' \
    KEY_NAME='${KEY_NAME:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' \
    nohup bash '$S/serve.sh' > /dev/null 2>&1 < /dev/null & echo '  rank $i launched on ${NODES[$i]}'"
done

echo "watch:    ssh ${NODES[0]} tail -f $ML4_LOG_DIR/ml4-rank0.log"
echo "endpoint: http://${NODES[0]}:$API_PORT   (served model: mistral-large-4)"
