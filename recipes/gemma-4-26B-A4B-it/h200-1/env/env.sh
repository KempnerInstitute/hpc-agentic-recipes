# Runtime environment for gemma-4-26B-A4B-it on one H200 GPU. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=gemma-4-26B-A4B-it-h200-1
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/h200-1/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

# The pip CUDA stack ships no nvcc and system gcc 8.5 is too old for the C++20 kernels, so load the
# cluster toolchain for any just-in-time compilation.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

# required: weights are staged locally, so never reach for the Hub at launch
export HF_HUB_OFFLINE=1
# required: unbuffered output, or the log stays empty during startup and looks hung
export PYTHONUNBUFFERED=1

# required: node-local JIT caches; a shared home hit stale file handles under concurrent compiles
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

# inherited: the DeepGEMM MoE path took an illegal memory access
# on GLM-5.2 and independently on Qwen3-Coder-480B-FP8 on this hardware. This checkpoint
# is an MoE, so the same kernels are in play; it was measured with the flag at 0 and untested at 1.
export VLLM_USE_DEEP_GEMM=0

# required: weight load plus torch.compile plus graph capture exceeds the 600s default readiness timeout
export VLLM_ENGINE_READY_TIMEOUT_S=3600
