# Runtime environment for gemma-4-26B-A4B-it on one RTX PRO 6000 Blackwell GPU. Source, do not execute.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=gemma-4-26B-A4B-it-rtx-1
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/rtx-1/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

export CUDA_HOME="${CUDA_HOME:-${CUDA13_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/rtx-1/cuda13}}"
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

export VLLM_ENGINE_READY_TIMEOUT_S=3600
