#!/usr/bin/env bash
# Join this node to the Ray cluster. Usage: ray_worker.sh <head_ip> <ray_port> <worker_ip>
set -euo pipefail
source "$(dirname "$0")/lib_env.sh"
HEAD_IP="${1:?head ip}"
RAY_PORT="${2:-6379}"
WORKER_IP="${3:?worker ip}"
ray stop >/dev/null 2>&1 || true
ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$WORKER_IP" --num-gpus=4 ${RAY_BLOCK:+--block}
