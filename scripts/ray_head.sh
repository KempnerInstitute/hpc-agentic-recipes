#!/usr/bin/env bash
# Start the Ray head on this node. Usage: ray_head.sh <head_ip> [ray_port]
set -euo pipefail
source "$(dirname "$0")/lib_env.sh"
HEAD_IP="${1:?head ip}"
RAY_PORT="${2:-6379}"
ray stop >/dev/null 2>&1 || true
ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-gpus=4 --disable-usage-stats ${RAY_BLOCK:+--block}
