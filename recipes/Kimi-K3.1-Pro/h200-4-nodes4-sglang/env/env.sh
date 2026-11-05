# Runtime environment for Kimi-K3.1-Pro on four H200 nodes, under SGLang in a container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_K31_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3.1-Pro/container/sglang-kimi-k31-pro.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Kimi-K3.1-Pro/container/sglang-kimi-k31-pro.sif" ]; then
    SIF="$MODELS_DIR/Kimi-K3.1-Pro/container/sglang-kimi-k31-pro.sif"
  else
    SIF="$_K31_SHARED"
  fi
fi

: "${K31_LOG_DIR:=/tmp/$USER/k31}"
: "${DIST_PORT:=29501}"

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_HUB_OFFLINE=1
export HF_HOME="$K31_LOG_DIR/hf"
export TRITON_CACHE_DIR="$K31_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$K31_LOG_DIR/inductor"
export PYTHONDONTWRITEBYTECODE=1
