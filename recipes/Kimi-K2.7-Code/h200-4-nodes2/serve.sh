#!/usr/bin/env bash
# Serve Kimi-K2.7-Code across two H200 nodes: TP4 inside each node, PP2 between them, over Ray.
#   bash recipes/Kimi-K2.7-Code/h200-4-nodes2/serve.sh <head_ip> [ray_port]
# Run this on the Ray head after both nodes have joined the cluster; serve.sbatch and serve_ssh.sh do
# that for you.
#
# The EXTRA_ARGS default is load-bearing, not a convenience. Without --skip-mm-profiling this launch
# completes weight loading and then hangs at multimodal profiling forever, at 0 percent GPU utilization,
# with no error. EXTRA_ARGS REPLACES the default rather than adding to it, so anything you pass must
# repeat those two flags unless you are deliberately reproducing the hang.
# ENFORCE_EAGER defaults to 1, matching the configuration the measured rate came from; CUDA graph
# capture has never been attempted for this model on this hardware. Pass ENFORCE_EAGER= to try it.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/env/env.sh"
MODEL="${MODEL:-$MODELS_DIR/Kimi-K2.7-Code}"
[ -d "$MODEL" ] || { echo "checkpoint not found: $MODEL   (set MODEL or MODELS_DIR)" >&2; exit 1; }
HEAD_IP="${1:-${RAY_HEAD_IP:-}}"
[ -n "$HEAD_IP" ] || { echo "usage: $(basename "$0") <head_ip> [ray_port]   (or set RAY_HEAD_IP)" >&2; exit 2; }
export RAY_ADDRESS="$HEAD_IP:${2:-${RAY_PORT:-6379}}"

EXTRA=()
# No colon in this expansion: an explicitly empty ENFORCE_EAGER must stay empty so graphs can be tried,
# while an unset one still gets the default of 1.
[ -n "${ENFORCE_EAGER-1}" ] && EXTRA+=(--enforce-eager)
[ -n "${ATTN_BACKEND:-}" ] && export VLLM_ATTENTION_BACKEND="$ATTN_BACKEND"
# A colon here, deliberately: an empty EXTRA_ARGS falls back to the multimodal flags rather than
# dropping them. Forgetting them costs an allocation to an indefinite hang, so empty means default.
EXTRA+=(${EXTRA_ARGS:---skip-mm-profiling --mm-processor-cache-gb 0})

exec vllm serve "$MODEL" \
  --served-model-name kimi-k2.7-code \
  --tensor-parallel-size "${TP:-4}" \
  --pipeline-parallel-size "${PP:-2}" \
  --distributed-executor-backend ray \
  --disable-custom-all-reduce \
  --trust-remote-code \
  --mm-encoder-tp-mode data \
  --host 0.0.0.0 --port "${API_PORT:-8000}" \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --enable-auto-tool-choice \
  --tool-call-parser "${TOOL_PARSER:-kimi_k2}" \
  --reasoning-parser "${REASONING_PARSER:-kimi_k2}" \
  "${EXTRA[@]}"
