# Runtime environment for gemma-4-31B-it on one RTX PRO 6000 Blackwell GPU. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=gemma-4-31B-it-rtx-1
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-31B-it/rtx-1/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

# required: the FlashInfer sm_120 JIT needs the complete conda CUDA 13.0 toolkit. The node's
# /usr/local/cuda-13 is runtime only and the pip nvcc wheels mix 13.0 and 13.2 between nvcc, cicc and
# ptxas, which breaks compilation.
export CUDA_HOME="${CUDA_HOME:-${CUDA13_DIR:-$ENV_ROOT/gemma-4-31B-it/rtx-1/cuda13}}"
export PATH="$CUDA_HOME/bin:$PATH"
# Compile time only, on purpose. The toolkit must not reach LD_LIBRARY_PATH: its libcudart shadows
# torch's CUDA 13 runtime and pulls in a libcupti.so.13 that is not installed, which breaks import.
# FlashInfer also compiles host C++ with g++, which does not add the CUDA include and library dirs
# by itself, so expose them here.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"

# CUDA 13 needs a modern host compiler; system gcc 8.5 is too old and gcc 13.2 is compatible.
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
# required: node-local FlashInfer workspace, for the same reason as the two caches above
export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/flashinfer"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true

# required: flashinfer-python is 0.6.15 for the kv_scale_format fix this engine needs, but no matching
# 0.6.15 cubin package exists, so flashinfer-cubin stays at 0.6.13 and the version check must be
# bypassed. The sm_120 kernels are then compiled from source on first launch.
export FLASHINFER_DISABLE_VERSION_CHECK=1

# required: weight load plus torch.compile plus graph capture exceeds the 600s default readiness timeout
export VLLM_ENGINE_READY_TIMEOUT_S=3600
