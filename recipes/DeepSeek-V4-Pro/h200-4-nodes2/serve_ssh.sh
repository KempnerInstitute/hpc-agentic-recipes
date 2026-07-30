#!/usr/bin/env bash
# Bring up DeepSeek-V4-Pro over SSH on two H200 nodes you already hold. Secondary path: prefer
# serve.sbatch. Reserved nodes are removed from the scheduler, which is why this exists.
#   bash recipes/DeepSeek-V4-Pro/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
# The endpoint runs on <head_node>. Both nodes must see the same repo checkout and the same
# environment path, since the Ray workers import vLLM from it.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
HEAD="${1:-${DSV4_H200_HEAD:-}}"
WORKER="${2:-${DSV4_H200_WORKER:-}}"
if [ -z "$HEAD" ] || [ -z "$WORKER" ]; then
  echo "usage: $(basename "$0") <head_node> <worker_node>   (or set DSV4_H200_HEAD and DSV4_H200_WORKER in common/site.conf)" >&2
  exit 2
fi
RAY_PORT="${RAY_PORT:-6379}"
LOG="$LOG_DIR/vllm-dsv4-h200.log"

ipof () { ssh -o BatchMode=yes "$1" 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"

ssh -o BatchMode=yes "$HEAD" "cd '$REPO_ROOT'; source '$S/env/env.sh'; ray stop >/dev/null 2>&1 || true; ray start --head --node-ip-address='$HEAD_IP' --port='$RAY_PORT' --num-gpus=4 --disable-usage-stats"
ssh -o BatchMode=yes "$WORKER" "cd '$REPO_ROOT'; source '$S/env/env.sh'; ray stop >/dev/null 2>&1 || true; ray start --address='$HEAD_IP:$RAY_PORT' --node-ip-address='$WORKER_IP' --num-gpus=4"

echo "launching deepseek-v4-pro on $HEAD:${API_PORT}"
ssh -o BatchMode=yes "$HEAD" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' PP='${PP:-}' PERF='${PERF:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/serve.sh' '$HEAD_IP' '$RAY_PORT' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $HEAD tail -f $LOG"
echo "endpoint: http://$HEAD:$API_PORT/v1   (served model: deepseek-v4-pro)"
