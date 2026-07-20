#!/usr/bin/env bash
# Bring up GLM-5.2-FP8 via SGLang across the two reserved nodes over SSH (TP=8, no PP, EAGLE/MTP).
# Keeps the vLLM path (serve_glm_ssh.sh) intact; this is the alternate engine.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$S")"
HEAD="${HEAD:-holygpu8a10101}"
WORKER="${WORKER:-holygpu8a10102}"
API_PORT="${API_PORT:-8000}"
mkdir -p "$REPO_DIR/logs"

ip_of () { ssh -o BatchMode=yes "$1" "ip -br -4 addr show ib0 | awk '{print \$3}' | cut -d/ -f1"; }
HEAD_IP="$(ip_of "$HEAD")"
echo "head=$HEAD ($HEAD_IP)  worker=$WORKER"

ENVS="MODEL='${MODEL:-}' API_PORT='$API_PORT' MAX_MODEL_LEN='${MAX_MODEL_LEN:-131072}' MEM_FRAC='${MEM_FRAC:-0.90}' NO_MTP='${NO_MTP:-}' SPEC_STEPS='${SPEC_STEPS:-}' SPEC_TOPK='${SPEC_TOPK:-}' SPEC_DRAFT='${SPEC_DRAFT:-}' DIST_PORT='${DIST_PORT:-20000}'"

echo "launching SGLang worker (rank 1) on $WORKER"
ssh -o BatchMode=yes "$WORKER" "cd '$REPO_DIR'; $ENVS nohup bash '$S/sglang_glm.sh' 1 '$HEAD_IP' > '$REPO_DIR/logs/sglang-glm-worker.log' 2>&1 < /dev/null & echo worker pid \$!"
echo "launching SGLang head (rank 0) on $HEAD"
ssh -o BatchMode=yes "$HEAD" "cd '$REPO_DIR'; $ENVS nohup bash '$S/sglang_glm.sh' 0 '$HEAD_IP' > '$REPO_DIR/logs/sglang-glm.log' 2>&1 < /dev/null & echo head pid \$!"
echo "watch:    ssh $HEAD tail -f $REPO_DIR/logs/sglang-glm.log"
echo "endpoint: http://$HEAD:$API_PORT/v1  (served model: glm-5.2, SGLang engine)"
