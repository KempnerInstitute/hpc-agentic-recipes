# Runtime environment for DeepSeek-V4-Pro on two RTX PRO 6000 nodes. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
source "$S/../../../../common/lib/api_key.sh"

VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

# required: the sm_120 just-in-time compiler needs a complete CUDA 13.0 toolkit, and the node's
# /usr/local/cuda-13 is runtime-only
export CUDA_HOME="${CUDA_HOME:-${CUDA13_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/cuda13}}"
export PATH="$CUDA_HOME/bin:$PATH"
# Compile-time only. Do not put the toolkit's libraries on LD_LIBRARY_PATH: its libcudart shadows
# torch's CUDA 13 runtime and pulls in a libcupti.so.13 that is not present, which breaks import.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

# CUDA 13 needs a modern host compiler; system gcc 8.5 is too old and gcc 13 works.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/13.2.0-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

# required: weights are staged locally, so never reach for the Hub at launch
export HF_HUB_OFFLINE=1
# required: unbuffered output, or the log stays empty during startup and looks hung
export PYTHONUNBUFFERED=1

# required: node-local JIT caches. Concurrent compiles from two nodes against a shared home hit stale
# file handles, and this is a two-node recipe, so the shared path is not an option.
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
# required for the same reason: FlashInfer compiles its sm_120 kernels here on first launch
export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

# required: flashinfer-python is 0.6.15 while no matching 0.6.15 cubin package exists, so
# flashinfer-cubin stays at 0.6.13 and the version check has to be bypassed. Kernels are then built
# from source on first launch, which is why the first request after a fresh environment is slow.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# required: RTX PRO 6000 nodes have no NVLink, and NCCL initialization hangs with no error on any
# multi-GPU job unless peer-to-peer is disabled
export NCCL_P2P_DISABLE=1

# required: weight load of 806 GiB across 16 ranks plus graph work exceeds the 600s default
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# verified: a holylfs06 OSS failover froze every rank on 2026-07-29 and PyTorch's heartbeat monitor
# killed two endpoints eight minutes later, at the 480s default, for a stall that later recovered
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

# inherited from the Hopper multi-node path, untested on RTX: pin the cross-node transport to the
# InfiniBand interface when the node exposes one, and leave NCCL's own selection in place when it does
# not, because whether these nodes present ib0 has not been checked on hardware.
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
# required: a silent NCCL failure across nodes is very hard to diagnose from the server log alone
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
