#!/usr/bin/env bash
# Bring up Kimi-K3.1-Pro over SSH on four H200 nodes you already hold.
#   bash recipes/Kimi-K3.1-Pro/h200-4-nodes4-sglang/serve_ssh.sh <n0> <n1> <n2> <n3>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 4 ] || { echo "usage: $(basename "$0") <node0> <node1> <node2> <node3>" >&2; exit 2; }
NODES=("$@")
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K3.1-Pro}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

HEAD_IB="$(ssh -o BatchMode=yes "${NODES[0]}" "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")"
[ -n "$HEAD_IB" ] || { echo "could not read ib0 on ${NODES[0]}" >&2; exit 1; }
echo "head ${NODES[0]} at $HEAD_IB, serving kimi-k3.1-pro on ${NODES[0]}:$API_PORT"

# Values are quoted with printf %q so an embedded apostrophe or space, for instance in a JSON
# EXTRA_ARGS, cannot break out of the remote command.
ENVS=""
for v in API_PORT DIST_PORT SIF MODELS_DIR MODEL MAX_MODEL_LEN MEM_FRACTION TP EP NNODES \
         KV_DTYPE MOE_RUNNER ATTN KEY_NAME KEY_FILE K31_LOG_DIR EXTRA_ARGS; do
  ENVS+=" $v=$(printf '%q' "${!v-}")"
done
ENVS+=" HEAD_IB=$(printf '%q' "$HEAD_IB")"

FAILED=0
for i in 0 1 2 3; do
  n="${NODES[$i]}"
  ssh -o BatchMode=yes "$n" "mkdir -p $(printf '%q' "$K31_LOG_DIR") && cd $(printf '%q' "$REPO_ROOT") && \
    RANK=$i$ENVS nohup bash $(printf '%q' "$S/serve.sh") > /dev/null 2>&1 < /dev/null & " \
    && echo "  rank $i launched on $n" \
    || { echo "  rank $i FAILED to launch on $n" >&2; FAILED=1; }
done
[ "$FAILED" = 0 ] || { echo "at least one rank did not launch; check the logs before retrying" >&2; exit 1; }

# A rank that dies during startup leaves nothing behind, so confirm each one is alive.
sleep 20
for i in 0 1 2 3; do
  n="${NODES[$i]}"
  if ssh -o BatchMode=yes "$n" "pgrep -u \$USER -f sglang_launch.py >/dev/null" 2>/dev/null; then
    echo "  rank $i alive on $n"
  else
    echo "  rank $i is NOT running on $n, see $K31_LOG_DIR/k31-rank$i.log" >&2; FAILED=1
  fi
done
[ "$FAILED" = 0 ] || exit 1

echo "watch:    ssh ${NODES[0]} tail -f $K31_LOG_DIR/k31-rank0.log"
echo "endpoint: http://${NODES[0]}:$API_PORT   (served model: kimi-k3.1-pro)"
