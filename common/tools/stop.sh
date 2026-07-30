#!/usr/bin/env bash
# Stop endpoints started over SSH, and wait for GPU memory to actually be released.
#
#   stop.sh <node> [node...]
#
# For a Slurm job use scancel instead.
#
# Releasing memory matters: vLLM holds the whole KV cache, and relaunching before the old process has
# exited fails with an out-of-memory error that looks like a sizing problem rather than a stale process.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"

[ $# -gt 0 ] || { echo "usage: $(basename "$0") <node> [node...]" >&2; exit 2; }

for node in "$@"; do
  echo "== $node"
  # Bracketed patterns cannot match this command's own line. A bare 'VLLM::' pattern also matches the
  # ssh command being run, so pkill would terminate its own session and return 255.
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" '
    pkill -9 -f "[v]llm serve" 2>/dev/null
    pkill -9 -f "VLLM::[EWA]" 2>/dev/null
    pkill -9 -f "[s]glang.launch_server" 2>/dev/null
    for i in $(seq 1 30); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
      procs=$(pgrep -cf "[v]llm serve" || true)
      [ "${used:-0}" -lt 512 ] && [ "${procs:-0}" = 0 ] && break
      sleep 2
    done
    echo "  remaining vllm processes: $(pgrep -cf "[v]llm serve" || echo 0)"
    echo "  gpu memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd" ")"
  ' 2>&1 | sed 's/^/  /'
done
