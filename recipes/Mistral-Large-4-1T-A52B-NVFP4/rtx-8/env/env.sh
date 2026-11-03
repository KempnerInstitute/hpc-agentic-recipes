# Runtime environment for Mistral-Large-4-1T-A52B-NVFP4 on one RTX node, under vLLM in a container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_ML4N_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Mistral-Large-4-1T-A52B-NVFP4/container/vllm-mistral-large-4.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4/container/vllm-mistral-large-4.sif" ]; then
    SIF="$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4/container/vllm-mistral-large-4.sif"
  elif [ -f "$MODELS_DIR/Mistral-Large-4-1T-A52B/container/vllm-mistral-large-4.sif" ]; then
    SIF="$MODELS_DIR/Mistral-Large-4-1T-A52B/container/vllm-mistral-large-4.sif"
  else
    SIF="$_ML4N_SHARED"
  fi
fi

: "${ML4N_LOG_DIR:=/tmp/$USER/mistral-large-4-nvfp4}"

export HF_HUB_OFFLINE=1
export HF_HOME="$ML4N_LOG_DIR/hf"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export VLLM_ENGINE_READY_TIMEOUT_S=6000
export PYTHONDONTWRITEBYTECODE=1
