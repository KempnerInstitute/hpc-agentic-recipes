#!/usr/bin/env bash
# Bring up Kimi-K3 over SSH on four H200 nodes you already hold. Secondary path: prefer serve.sbatch.
#   bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>
#
# node0 is the head: every rank dials its InfiniBand address to form the 16-rank group, so that
# address is read from the node rather than assumed.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 4 ] || { echo "usage: $(basename "$0") <node0> <node1> <node2> <node3>" >&2; exit 2; }
NODES=("$@")
API_PORT="${API_PORT:-8000}"

command -v singularity >/dev/null 2>&1 || echo "note: singularity is resolved on the compute nodes, not here" >&2
[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; echo "run env/build.sh first" >&2; exit 1; }

HEAD_IB="$(ssh -o BatchMode=yes "${NODES[0]}" "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")"
[ -n "$HEAD_IB" ] || { echo "could not read ib0 on ${NODES[0]}" >&2; exit 1; }
echo "head ${NODES[0]} at $HEAD_IB, serving kimi-k3 on ${NODES[0]}:$API_PORT"

for i in 0 1 2 3; do
  ssh -o BatchMode=yes "${NODES[$i]}" "mkdir -p '$K3_LOG_DIR'; cd '$REPO_ROOT'; \
    RANK=$i HEAD_IB='$HEAD_IB' API_PORT='$API_PORT' SPEC_MODE='${SPEC_MODE:-none}' \
    MODEL='${MODEL:-}' DRAFT='${DRAFT:-}' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' \
    MEM_FRACTION='${MEM_FRACTION:-}' MAMBA_RATIO='${MAMBA_RATIO:-}' \
    MAMBA_CACHE_STRATEGY='${MAMBA_CACHE_STRATEGY:-}' WIDE='${WIDE:-0}' SIF='$SIF' \
    nohup bash '$S/serve.sh' > /dev/null 2>&1 < /dev/null & echo '  rank $i launched on ${NODES[$i]}'"
done

echo "watch:    ssh ${NODES[0]} tail -f $K3_LOG_DIR/k3-rank0.log"
echo "endpoint: http://${NODES[0]}:$API_PORT/v1   (served model: kimi-k3, OpenAI API only)"
echo "startup:  14 to 16 minutes, dominated by reading 1.5 TiB of weights"
