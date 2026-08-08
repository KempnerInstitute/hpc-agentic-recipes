# Runtime environment for Kimi-K3 on four H200 nodes. Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_K3_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3/container/sglang-kimi-k3-cu12.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Kimi-K3/container/sglang-kimi-k3-cu12.sif" ]; then
    SIF="$MODELS_DIR/Kimi-K3/container/sglang-kimi-k3-cu12.sif"
  else
    SIF="$_K3_SHARED"
  fi
fi

: "${K3_LOG_DIR:=/tmp/$USER/k3}"

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

export TRITON_CACHE_DIR="$K3_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$K3_LOG_DIR/inductor"

export HF_HUB_OFFLINE=1
export HF_HOME="$K3_LOG_DIR/hf"

export PYTHONDONTWRITEBYTECODE=1
