# Runtime environment for Kimi-K2.7-Code on two H200 nodes. Source, do not execute.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Kimi-K2.7-Code/h200-4-nodes2/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

export VLLM_USE_DEEP_GEMM=0

export VLLM_ENGINE_READY_TIMEOUT_S=3600

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

_hca=""
for _d in /sys/class/infiniband/*; do
  _n="$(basename "$_d")"
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  [[ "$_s" == *ACTIVE* && "$_l" == InfiniBand* ]] && _hca+="${_hca:+,}$_n"
done
export NCCL_IB_HCA="$_hca"
unset _hca _d _n _s _l

:
