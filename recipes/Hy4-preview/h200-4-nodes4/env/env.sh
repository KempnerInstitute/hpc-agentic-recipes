# Runtime environment for Hy4-preview on four H200 nodes, under SGLang in a container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_HY4_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Hy4-preview/container/sglang-hy4-preview.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Hy4-preview/container/sglang-hy4-preview.sif" ]; then
    SIF="$MODELS_DIR/Hy4-preview/container/sglang-hy4-preview.sif"
  else
    SIF="$_HY4_SHARED"
  fi
fi

: "${HY4_LOG_DIR:=/tmp/$USER/hy4}"
: "${DIST_PORT:=29501}"

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_HUB_OFFLINE=1
export HF_HOME="$HY4_LOG_DIR/hf"
export TRITON_CACHE_DIR="$HY4_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$HY4_LOG_DIR/inductor"
export PYTHONDONTWRITEBYTECODE=1
