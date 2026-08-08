#!/usr/bin/env bash
# Join a Ray worker to the head for the two-node Kimi-K2.7-Code recipe.
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
