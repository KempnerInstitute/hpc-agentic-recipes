#!/usr/bin/env bash
# Bring up Muse-Spark-Open-816B-A42B over SSH on two nodes you already hold.
#   bash recipes/Muse-Spark-Open-816B-A42B/h200-4-nodes4/serve_ssh.sh <head_node> <worker_node>
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../common/defaults.sh"
if [ "$#" -ge 2 ]; then
  HEAD="$1"; shift; WORKERS=("$@")
else
  HEAD="${MUSE_HEAD:-}"; read -r -a WORKERS <<< "${MUSE_WORKERS:-}"
fi
if [ -z "$HEAD" ] || [ "${#WORKERS[@]}" -eq 0 ]; then
  echo "usage: $(basename "$0") <head_node> <worker_node>...   (or set MUSE_HEAD and MUSE_WORKERS in common/site.conf)" >&2
  exit 2
fi
RAY_PORT="${RAY_PORT:-6379}"
LOG="$LOG_DIR/vllm-muse-fp8.log"

ipof () { ssh -o BatchMode=yes "$1" 'ip -br -4 addr show ib0 | awk "{print \$3}" | cut -d/ -f1'; }
HEAD_IP="$(ipof "$HEAD")"
[ -n "$HEAD_IP" ] || { echo "could not read an ib0 address from $HEAD" >&2; exit 1; }
echo "head=$HEAD ($HEAD_IP)  workers=${WORKERS[*]}"

ssh -o BatchMode=yes "$HEAD" "bash '$S/ray_head.sh' '$HEAD_IP' '$RAY_PORT'"
for w in "${WORKERS[@]}"; do
  w_ip="$(ipof "$w")"
  [ -n "$w_ip" ] || { echo "could not read an ib0 address from $w" >&2; exit 1; }
  ssh -o BatchMode=yes "$w" "bash '$S/ray_worker.sh' '$HEAD_IP' '$RAY_PORT' '$w_ip'"
done
ssh -o BatchMode=yes "$HEAD" "source '$S/env/env.sh'; python -c 'import ray; ray.init(address=\"auto\"); print(\"cluster GPUs:\", ray.cluster_resources().get(\"GPU\"))'"

echo "launching muse-spark on $HEAD:${API_PORT}"
ssh -o BatchMode=yes "$HEAD" "mkdir -p '$LOG_DIR'; cd '$REPO_ROOT'; MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-}' GPU_UTIL='${GPU_UTIL:-}' KV_DTYPE='${KV_DTYPE:-}' MTP_TOKENS='${MTP_TOKENS:-}' TP='${TP:-}' PP='${PP:-}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' CUDAGRAPH_MODE='${CUDAGRAPH_MODE:-}' TOOL_PARSER='${TOOL_PARSER:-}' REASONING_PARSER='${REASONING_PARSER:-}' EXTRA_ARGS='${EXTRA_ARGS:-}' nohup bash '$S/serve.sh' '$HEAD_IP' '$RAY_PORT' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $HEAD tail -f $LOG"
echo "endpoint: http://$HEAD:$API_PORT/v1   (served model: muse-spark)"
