#!/usr/bin/env bash
# Bring up Kimi-K2.7-Code across two H200 nodes over SSH (Ray + vLLM, TP=4 x PP=2).
# TP stays intra-node (NVLink all-reduce); PP crosses nodes. Pure TP8 across nodes hangs at NCCL init,
# and cross-node multimodal profiling deadlocks, so EXTRA_ARGS defaults to --skip-mm-profiling.
# Configurable inputs (defaults in scripts/config.sh): KIMI_HEAD, KIMI_WORKER, MODEL, API_PORT, RAY_PORT.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
HEAD="$KIMI_HEAD"
WORKER="$KIMI_WORKER"
MODEL="${MODEL:-$KIMI_MODEL}"
RAY_PORT="${RAY_PORT:-6379}"
mkdir -p "$REPO_DIR/logs"

ip_of () { ssh -o BatchMode=yes "$1" "ip -br -4 addr show ib0 | awk '{print \$3}' | cut -d/ -f1"; }
HEAD_IP="$(ip_of "$HEAD")"
WORKER_IP="$(ip_of "$WORKER")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER ($WORKER_IP)  model: $MODEL"

ssh -o BatchMode=yes "$HEAD" "bash '$S/ray_head.sh' '$HEAD_IP' '$RAY_PORT'"
ssh -o BatchMode=yes "$WORKER" "bash '$S/ray_worker.sh' '$HEAD_IP' '$RAY_PORT' '$WORKER_IP'"
ssh -o BatchMode=yes "$HEAD" "source '$S/lib_env.sh'; python -c 'import ray; ray.init(address=\"auto\"); print(\"cluster GPUs:\", ray.cluster_resources().get(\"GPU\"))'"

echo "launching vllm on $HEAD (log: logs/vllm-kimi-h200.log)"
ssh -o BatchMode=yes "$HEAD" "cd '$REPO_DIR'; ENV_LIB=lib_env.sh RAY=1 MODEL='$MODEL' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-32768}' GPU_UTIL='${GPU_UTIL:-0.90}' TP='${TP:-4}' PP='${PP:-2}' ENFORCE_EAGER='${ENFORCE_EAGER:-1}' EXTRA_ARGS='${EXTRA_ARGS:---skip-mm-profiling --mm-processor-cache-gb 0}' nohup bash '$S/vllm_kimi.sh' '$HEAD_IP' '$RAY_PORT' > '$REPO_DIR/logs/vllm-kimi-h200.log' 2>&1 < /dev/null & echo launched pid \$!"
echo "watch:    ssh $HEAD tail -f $REPO_DIR/logs/vllm-kimi-h200.log"
echo "endpoint: http://$HEAD:$API_PORT/v1  (served model: kimi-k2.7-code)"
