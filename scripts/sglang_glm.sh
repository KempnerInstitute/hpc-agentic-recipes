#!/usr/bin/env bash
# Serve GLM-5.2-FP8 via SGLang across two nodes (TP=8, no pipeline parallelism) with EAGLE/MTP
# speculative decoding. Usage: sglang_glm.sh <node_rank> <dist_init_host>
# SGLang can use GLM-5.2's MTP head (vLLM cannot, because that needs PP to span two nodes).
set -euo pipefail
source "$(dirname "$0")/lib_env_sglang.sh"
NODE_RANK="${1:?node rank (0=head, 1=worker)}"
DIST_HOST="${2:?dist-init host/ip}"
MODEL="${MODEL:?set MODEL (launch via serve_sglang_glm_ssh.sh)}"
SPEC=()
if [ -z "${NO_MTP:-}" ]; then
  SPEC+=(--speculative-algorithm EAGLE
         --speculative-num-steps "${SPEC_STEPS:-5}"
         --speculative-eagle-topk "${SPEC_TOPK:-1}"
         --speculative-num-draft-tokens "${SPEC_DRAFT:-6}")
fi
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name glm-5.2 \
  --tp 8 \
  --nnodes 2 \
  --node-rank "$NODE_RANK" \
  --dist-init-addr "${DIST_HOST}:${DIST_PORT:-20000}" \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --context-length "${MAX_MODEL_LEN:-131072}" \
  --mem-fraction-static "${MEM_FRAC:-0.90}" \
  --trust-remote-code \
  --reasoning-parser "${REASONING_PARSER:-glm45}" \
  --tool-call-parser "${TOOL_PARSER:-glm47}" \
  "${SPEC[@]}"
