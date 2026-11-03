# Runtime environment for Mistral-Large-4-1T-A52B FP8 on four H200 nodes, under SGLang in a container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_ML4_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Mistral-Large-4-1T-A52B/container/sglang-mistral-large-4.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Mistral-Large-4-1T-A52B/container/sglang-mistral-large-4.sif" ]; then
    SIF="$MODELS_DIR/Mistral-Large-4-1T-A52B/container/sglang-mistral-large-4.sif"
  elif [ -f "$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4/container/sglang-mistral-large-4.sif" ]; then
    SIF="$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4/container/sglang-mistral-large-4.sif"
  else
    SIF="$_ML4_SHARED"
  fi
fi

: "${ML4_LOG_DIR:=/tmp/$USER/mistral-large-4}"
: "${DIST_PORT:=29500}"

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_HUB_OFFLINE=1
export HF_HOME="$ML4_LOG_DIR/hf"
export TRITON_CACHE_DIR="$ML4_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$ML4_LOG_DIR/inductor"
export PYTHONDONTWRITEBYTECODE=1
