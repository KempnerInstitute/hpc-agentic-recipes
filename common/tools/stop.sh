#!/usr/bin/env bash
# Stop endpoints started over SSH, and wait until the GPUs and the host are actually clear.
#
#   stop.sh <node> [node...]
#
# For a Slurm job use scancel instead, and nothing else. The multi-node sbatch recipes request every node
# they need with --nodes and start Ray through srun inside that allocation, so Slurm's cgroups own every
# process on every node and tear all of them down when the job ends. This script exists because the SSH
# path has no such owner: it starts Ray with a plain ssh, so when the endpoint goes away the Ray daemons
# stay, and only something like this removes them.
#
# Two things need clearing, not one. vLLM holds the whole KV cache, so relaunching before the old process
# exits fails with an out-of-memory error that reads like a sizing mistake rather than a stale process. Ray
# separately leaves a GCS server and a set of dashboard workers that hold no GPU memory at all, so they
# survive any check that only looks at nvidia-smi while still occupying several GB of host RAM each.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"

[ $# -gt 0 ] || { echo "usage: $(basename "$0") <node> [node...]" >&2; exit 2; }

rc=0
for node in "$@"; do
  echo "== $node"
  # Every pattern below is bracketed, and that is load-bearing twice over. It stops pkill matching the ssh
  # command being run, which would kill its own session and return 255. It also stops the counting grep at
  # the end matching this script's own command line: an unbracketed "VLLM::" there made a clean node report
  # one leftover process forever.
  ssh -o BatchMode=yes -o ConnectTimeout=15 "$node" '
    me=$$; parent=$PPID
    pkill -9 -f "[v]llm serve" 2>/dev/null
    pkill -9 -f "[V]LLM::[EWA]" 2>/dev/null
    pkill -9 -f "[s]glang.launch_server" 2>/dev/null

    # Ask Ray to shut down properly first. ray lives in the recipe venv rather than on PATH, so find it
    # from a process already running instead of guessing which recipe launched this endpoint. Assuming ray
    # was on PATH is exactly why Ray outlived an earlier teardown.
    V="$(ps -u "$USER" -o args= 2>/dev/null | grep -oE "/[^ ]*/venv" | sort -u | head -1)"
    if [ -n "$V" ] && [ -x "$V/bin/ray" ]; then "$V/bin/ray" stop --force >/dev/null 2>&1; fi
    sleep 3

    # Whatever ignored that gets killed by explicit PID rather than by pkill, so a bad pattern can never
    # take out this shell. The patterns are bracketed for a second reason: grep in the pipeline below has
    # the pattern text on its own command line, and unbracketed it would match itself and be counted.
    for p in $(ps -u "$USER" -o pid=,args= 2>/dev/null \
               | grep -E "[r]aylet|[g]cs_server|[r]ay-dashboard|[p]lasma_store|[r]ay::|/[r]ay/" \
               | awk "{print \$1}"); do
      [ "$p" = "$me" ] || [ "$p" = "$parent" ] || kill -9 "$p" 2>/dev/null
    done

    # Last resort for anything still holding device memory, whatever it calls itself.
    for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -9 "$p" 2>/dev/null
    done

    for i in $(seq 1 40); do
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
      apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
      [ "${used:-0}" -lt 512 ] && [ "${apps:-0}" = 0 ] && break
      sleep 2
    done

    # Report what is left rather than assuming success. Bracketed patterns keep the pipeline from counting
    # its own grep, and the PID checks drop this shell and its parent.
    left=0
    for p in $(ps -u "$USER" -o pid=,args= 2>/dev/null \
               | grep -E "[v]llm serve|[V]LLM::|[s]glang.launch_server|[r]aylet|[g]cs_server|[r]ay-dashboard|[p]lasma_store" \
               | awk "{print \$1}"); do
      [ "$p" = "$me" ] || [ "$p" = "$parent" ] || left=$((left + 1))
    done
    echo "  gpu memory in use: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd" ")"
    echo "  gpu compute processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)"
    echo "  serving or ray processes left: $left"
    [ "$left" = 0 ]
  ' 2>&1 | sed "s/^/  /" || { echo "  WARNING: $node is not clean, run this again or inspect by hand"; rc=1; }
done
exit "$rc"
