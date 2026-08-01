#!/usr/bin/env bash
# Start the Ray head on this node, for GLM-5.2-FP8 across two H200 nodes.
#   bash recipes/GLM-5.2-FP8/h200-4-nodes2/ray_head.sh <head_ip> [ray_port]
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
NUM_GPUS="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"

ray stop >/dev/null 2>&1 || true
exec ray start --head \
  --node-ip-address="$HEAD_IP" \
  --port="$RAY_PORT" \
  --num-gpus="$NUM_GPUS" \
  --disable-usage-stats ${RAY_BLOCK:+--block}
