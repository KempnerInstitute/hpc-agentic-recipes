#!/usr/bin/env bash
# Join this node to the Ray cluster, for Kimi-K2.7-Code across two H200 nodes.
#   bash recipes/Kimi-K2.7-Code/h200-4-nodes2/ray_worker.sh <head_ip> <ray_port> <worker_ip>
# Set RAY_BLOCK=1 to stay in the foreground, which is what serve.sbatch needs so Slurm keeps the step
# alive. Leave it unset for the SSH path, where ray start daemonizes and the shell returns.
#
# The GPU count comes from the allocation rather than a hardcoded value, so a node with a different
# GPU count is neither under- nor over-subscribed; 4 is the H200 default. Environment settings come
# from this recipe's own env/env.sh.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
HEAD_IP="${1:?head ip}"
RAY_PORT="${2:-${RAY_PORT:-6379}}"
WORKER_IP="${3:?worker ip}"
NUM_GPUS="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

ray stop >/dev/null 2>&1 || true
exec ray start \
  --address="$HEAD_IP:$RAY_PORT" \
  --node-ip-address="$WORKER_IP" \
  --num-gpus="$NUM_GPUS" ${RAY_BLOCK:+--block}
