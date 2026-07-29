#!/usr/bin/env bash
# Bring up GLM-5.2 FP8 across two H200 nodes over SSH (Ray + vLLM, TP=4 x PP=2).
# Configurable inputs (defaults in scripts/config.sh): GLM52_HEAD, GLM52_WORKER, MODEL, API_PORT, RAY_PORT.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
HEAD="$GLM52_HEAD"
WORKER="$GLM52_WORKER"
MODEL="${MODEL:-$GLM52_MODEL}"
RAY_PORT="${RAY_PORT:-6379}"

LOG="$LOG_DIR/vllm-glm.log"

ip_of () { ssh -o BatchMode=yes "$1" "ip -br -4 addr show ib0 | awk '{print \$3}' | cut -d/ -f1"; }
HEAD_IP="$(ip_of "$HEAD")"
WORKER_IP="$(ip_of "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)  model: $MODEL"

ssh -o BatchMode=yes "$HEAD" "bash '$S/ray_head.sh' '$HEAD_IP' '$RAY_PORT'"
ssh -o BatchMode=yes "$WORKER" "bash '$S/ray_worker.sh' '$HEAD_IP' '$RAY_PORT' '$WORKER_IP'"
ssh -o BatchMode=yes "$HEAD" "source '$S/lib_env.sh'; python -c 'import ray; ray.init(address=\"auto\"); print(\"cluster GPUs:\", ray.cluster_resources().get(\"GPU\"))'"

echo "launching vllm on $HEAD (log: $LOG)"
ssh -o BatchMode=yes "$HEAD" "cd '$REPO_DIR'; mkdir -p '$LOG_DIR'; MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-131072}' GPU_UTIL='${GPU_UTIL:-0.90}' ENFORCE_EAGER='${ENFORCE_EAGER:-}' NO_MTP='${NO_MTP:-}' PERF='${PERF:-}' nohup bash '$S/vllm_glm.sh' '$HEAD_IP' '$RAY_PORT' > '$LOG' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $HEAD tail -f $LOG"
echo "endpoint: http://$HEAD:$API_PORT/v1  (served model: glm-5.2)"
