# Runtime environment for DeepSeek-V4-Pro on two RTX PRO 6000 nodes. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=DeepSeek-V4-Pro-rtx-8-nodes2
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

export CUDA_HOME="${CUDA_HOME:-${CUDA13_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/cuda13}}"
export PATH="$CUDA_HOME/bin:$PATH"
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/13.2.0-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

export FLASHINFER_DISABLE_VERSION_CHECK=1

export NCCL_P2P_DISABLE=1

export VLLM_ENGINE_READY_TIMEOUT_S=3600

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

if [ -d /sys/class/net/ib0 ]; then
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ib0}"
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-ib0}"
  _hca=""
  for _d in /sys/class/infiniband/*; do
    _n="$(basename "$_d")"
    _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
    _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
    [[ "$_s" == *ACTIVE* && "$_l" == InfiniBand* ]] && _hca+="${_hca:+,}$_n"
  done
  [ -n "$_hca" ] && export NCCL_IB_HCA="$_hca"
  unset _hca _d _n _s _l
fi
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
