#!/usr/bin/env bash
# Bring up Hy4-preview over SSH on four H200 nodes you already hold.
#   bash recipes/Hy4-preview/h200-4-nodes4/serve_ssh.sh <n0> <n1> <n2> <n3>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"

[ $# -eq 4 ] || { echo "usage: $(basename "$0") <node0> <node1> <node2> <node3>" >&2; exit 2; }
NODES=("$@")
API_PORT="${API_PORT:-8000}"
MODEL="${MODEL:-$MODELS_DIR/Hy4-preview}"

[ -f "$SIF" ] || { echo "container not found: $SIF" >&2; exit 1; }
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL" >&2; exit 1; }

# These are whole-node allocations, so refuse any node whose GPUs are already busy rather than
# contending with another user's work.
for n in "${NODES[@]}"; do
  BUSY="$(timeout 30 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" \
    "set -o pipefail; nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")" || {
    echo "could not read GPU state on $n; refusing to launch blind" >&2
    exit 1
  }
  case "$BUSY" in
    ''|*[!0-9]*) echo "unreadable GPU count from $n: '$BUSY'" >&2; exit 1 ;;
  esac
  if [ "$BUSY" != 0 ]; then
    echo "$n already has ${BUSY} GPU compute process(es); refusing to launch" >&2
    exit 1
  fi
done

HEAD_IB="$(timeout 60 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "${NODES[0]}" "ip -o -4 addr show ib0 | awk '{print \$4}' | cut -d/ -f1")"
[ -n "$HEAD_IB" ] || { echo "could not read ib0 on ${NODES[0]}" >&2; exit 1; }
echo "head ${NODES[0]} at $HEAD_IB, serving hy4-preview on ${NODES[0]}:$API_PORT"

q() { printf '%s' "${1//\'/\'\\\'\'}"; }

FAILED=0
for i in 0 1 2 3; do
  n="${NODES[$i]}"
  # q() escapes single quotes so a value containing an apostrophe cannot break out of the remote
  # single quoting. The assignments are written out explicitly so the repo's forwarding check can
  # see every tunable that serve.sh reads.
  # The whole list runs in a group redirected to /dev/null: backgrounding an AND-list alone spawns a
  # subshell that still holds the ssh channel, and ssh would then wait for the rank to exit.
  timeout 120 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" \
    "{ mkdir -p '$(q "$HY4_LOG_DIR")' && cd '$(q "$REPO_ROOT")' && \
       RANK=$i HEAD_IB='$(q "$HEAD_IB")' API_PORT='$(q "${API_PORT:-}")' \
       DIST_PORT='$(q "${DIST_PORT:-}")' SIF='$(q "$SIF")' MODELS_DIR='$(q "$MODELS_DIR")' \
       MODEL='$(q "$MODEL")' MAX_MODEL_LEN='$(q "${MAX_MODEL_LEN:-}")' \
       MEM_FRACTION='$(q "${MEM_FRACTION:-}")' TP='$(q "${TP:-}")' \
       NNODES='$(q "${NNODES:-}")' MOE_RUNNER='$(q "${MOE_RUNNER:-}")' ATTN='$(q "${ATTN:-}")' \
       SPEC_MODE='$(q "${SPEC_MODE:-}")' SPEC_STEPS='$(q "${SPEC_STEPS:-}")' \
       SPEC_TOPK='$(q "${SPEC_TOPK:-}")' SPEC_DRAFT='$(q "${SPEC_DRAFT:-}")' \
       TOOL_PARSER='$(q "${TOOL_PARSER:-}")' REASONING_PARSER='$(q "${REASONING_PARSER:-}")' \
       HY4_LOAD_BARRIER_PATCH='$(q "${HY4_LOAD_BARRIER_PATCH:-}")' \
       KEY_NAME='$(q "${KEY_NAME:-}")' KEY_FILE='$(q "${KEY_FILE:-}")' \
       HY4_LOG_DIR='$(q "$HY4_LOG_DIR")' EXTRA_ARGS='$(q "${EXTRA_ARGS:-}")' \
       setsid bash '$(q "$S/serve.sh")' ; } > /dev/null 2>&1 &" \
    && echo "  rank $i launched on $n" \
    || { echo "  rank $i FAILED to launch on $n" >&2; FAILED=1; }
done

[ "$FAILED" = 0 ] || { echo "at least one rank did not launch; check the logs before retrying" >&2; exit 1; }

# A rank that dies during startup leaves nothing behind, so confirm each one is alive.
sleep 20
for i in 0 1 2 3; do
  n="${NODES[$i]}"
  if timeout 45 ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$n" \
       "p=\$(cat '$(q "$HY4_LOG_DIR")/hy4-rank$i.pid' 2>/dev/null); [ -n \"\$p\" ] && kill -0 \"\$p\" 2>/dev/null && tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -qE 'sglang_launch.py|Singularity runtime parent'" 2>/dev/null; then
    echo "  rank $i alive on $n"
  else
    echo "  rank $i is NOT running on $n, see $HY4_LOG_DIR/hy4-rank$i.log" >&2; FAILED=1
  fi
done
[ "$FAILED" = 0 ] || exit 1

echo "watch:    ssh ${NODES[0]} tail -f $HY4_LOG_DIR/hy4-rank0.log"
echo "endpoint: http://${NODES[0]}:$API_PORT   (served model: hy4-preview)"
