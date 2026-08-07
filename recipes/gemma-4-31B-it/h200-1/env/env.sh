# Runtime environment for gemma-4-31B-it on one H200 GPU. Source, do not execute.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=gemma-4-31B-it-h200-1
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-31B-it/h200-1/venv}"
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
