#!/usr/bin/env bash
# Start the Ray head for the two-node Kimi-K2.7-Code recipe.
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
