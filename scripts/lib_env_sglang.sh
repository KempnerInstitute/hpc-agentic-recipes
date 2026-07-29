# Shared runtime environment for multi-node SGLang (source this, do not execute).
# Separate venv from vLLM (.venv-sglang) so both engines coexist.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
# Set VENV_DIR to run from a copy of the environment on faster storage. Startup is dominated by
# page-faulting the torch shared objects and stat-ing the many small package files.
source "${VENV_DIR:-$REPO_DIR/.venv-sglang}/bin/activate"
# The pip CUDA stack ships no nvcc, and system gcc 8.5 is too old for the FP8/DSA C++20 kernels;
# load the cluster CUDA 12.9 + gcc 12.2 toolchain so nvcc JIT works.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# Node-local JIT cache; shared NFS home hit stale file handles under concurrent multi-node compiles.
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true
# PyTorch kills the process when the NCCL watchdog thread stops sending heartbeats, on the assumption
# that the collective hung. A stalled network filesystem freezes every rank the same way, so at the
# 480s default a storage outage that later recovers still takes the endpoint down permanently.
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
