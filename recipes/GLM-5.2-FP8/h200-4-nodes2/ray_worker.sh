#!/usr/bin/env bash
# Join this node to the Ray cluster, for GLM-5.2-FP8 across two H200 nodes.
#   bash recipes/GLM-5.2-FP8/h200-4-nodes2/ray_worker.sh <head_ip> <ray_port> <worker_ip>
# Set RAY_BLOCK=1 to stay in the foreground, which is what serve.sbatch needs so Slurm keeps the step
# alive. Leave it unset for the SSH path, where ray start daemonizes and the shell returns.
#
# Two changes from the pre-restructure scripts/ray_worker.sh, both required by the restructure: it
# sourced a sibling lib_env.sh, which no longer exists now that each recipe carries its own
# environment, and it hardcoded --num-gpus=4, which silently under- or over-subscribed any node with a
# different GPU count. The count now comes from the allocation, with 4 as the H200 default.
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
