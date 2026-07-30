# Runtime environment for Qwen3-235B-A22B on one RTX PRO 6000 Blackwell node. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
source "$S/../../../../common/lib/api_key.sh"

VENV="${VENV_DIR:-$ENV_ROOT/Qwen3-235B-A22B/rtx-8/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

CUDA13="${CUDA13_DIR:-$ENV_ROOT/Qwen3-235B-A22B/rtx-8/cuda13}"
# The node's /usr/local/cuda-13 is runtime-only, and the pip nvcc wheels mix 13.0 and 13.2 between
# nvcc, cicc and ptxas, which breaks the JIT, so this recipe builds its own toolkit through conda.
# required: FlashInfer compiles the sm_120 kernels from source at first launch and needs that toolkit.
export CUDA_HOME="$CUDA13"
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "warning: no nvcc at $CUDA_HOME/bin; run env/build.sh first" >&2
export PATH="$CUDA_HOME/bin:$PATH"

# Compile-time exposure only. Do not add the toolkit's libraries to LD_LIBRARY_PATH: its libcudart
# shadows torch's CUDA 13 runtime and pulls in a libcupti.so.13 that is not present, which breaks
# import entirely. FlashInfer also compiles host C++ with g++, which does not pick up the CUDA
# include and library directories on its own, so they are exposed here instead.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

# CUDA 13 needs a modern host compiler and system gcc 8.5 is too old for the C++20 kernels, so load
# the cluster toolchain for the just-in-time compilation.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/13.2.0-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

# required: weights are staged locally, so never reach for the Hub at launch
export HF_HUB_OFFLINE=1
# required: unbuffered output, or the log stays empty during startup and looks hung
export PYTHONUNBUFFERED=1

# required: node-local JIT caches; a shared home hit stale file handles under concurrent compiles
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

# verified: flashinfer-python is 0.6.15, which accepts the kv_scale_format argument vLLM 0.25.1's
# sm_120 attention backend passes, but no flashinfer-cubin 0.6.15 exists, so the version check has
# to be bypassed and the kernels are compiled from source on first launch, 2026-07-19
export FLASHINFER_DISABLE_VERSION_CHECK=1

# verified: RTX PRO 6000 has no NVLink, so peer-to-peer over PCIe must be off or NCCL init hangs with
# no error and the server never becomes ready, 2026-07-19
export NCCL_P2P_DISABLE=1

# required: weight load plus first-time JIT plus graph capture exceeds the 600s default timeout
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# verified: a holylfs06 OSS failover froze every rank on 2026-07-29 and PyTorch's heartbeat monitor
# killed two endpoints eight minutes later, at the 480s default, for a stall that later recovered
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"
