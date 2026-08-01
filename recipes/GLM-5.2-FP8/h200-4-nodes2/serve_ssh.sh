#!/usr/bin/env bash
# Bring up GLM-5.2-FP8 over SSH on two H200 nodes you already hold. Secondary path: prefer
# serve.sbatch.
#   bash recipes/GLM-5.2-FP8/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
# The endpoint runs on <head_node>. Both nodes must see the same repo checkout and the same environment
# path, since the Ray workers import vLLM from it.
#
# No hostname is hardcoded anywhere: the node names are arguments, and each node's ib0 address is read
# from the node itself, because that is the interface env/env.sh pins the collectives to.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
HEAD="${1:-${GLM52_HEAD:-}}"
WORKER="${2:-${GLM52_WORKER:-}}"
if [ -z "$HEAD" ] || [ -z "$WORKER" ]; then
  echo "usage: $(basename "$0") <head_node> <worker_node>   (or set GLM52_HEAD and GLM52_WORKER in common/site.conf)" >&2
  exit 2
fi
RAY_PORT="${RAY_PORT:-6379}"
LOG="$LOG_DIR/vllm-glm52-fp8.log"

ipof () { ssh -o BatchMode=yes "$1" 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
WORKER_IP="$(ipof "$WORKER")"
[ -n "$HEAD_IP" ] || { echo "could not read an ib0 address from $HEAD" >&2; exit 1; }
[ -n "$WORKER_IP" ] || { echo "could not read an ib0 address from $WORKER" >&2; exit 1; }
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)"

ssh -o BatchMode=yes "$HEAD" "bash '$S/ray_head.sh' '$HEAD_IP' '$RAY_PORT'"
ssh -o BatchMode=yes "$WORKER" "bash '$S/ray_worker.sh' '$HEAD_IP' '$RAY_PORT' '$WORKER_IP'"
# Confirm the cluster really has 8 GPUs before loading 756 GB of weights. A worker that failed to join
# otherwise shows up much later as an engine that waits for resources it will never get.
ssh -o BatchMode=yes "$HEAD" "source '$S/env/env.sh'; python -c 'import ray; ray.init(address=\"auto\"); print(\"cluster GPUs:\", ray.cluster_resources().get(\"GPU\"))'"

echo "launching glm-5.2 on $HEAD:${API_PORT}"
ssh -o BatchMode=yes "$HEAD" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' TP='${TP:-}' PP='${PP:-}' PERF='${PERF:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/serve.sh' '$HEAD_IP' '$RAY_PORT' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $HEAD tail -f $LOG"
echo "endpoint: http://$HEAD:$API_PORT/v1   (served model: glm-5.2)"
