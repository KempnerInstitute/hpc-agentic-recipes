# Runtime environment for Muse-Glimmer-30B on one H200 GPU. Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Muse-Glimmer-30B/h200-1/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

# FlashInfer compiles its sampling kernel at first launch and needs nvcc on PATH. Without the module
# load the build fails with ninja exit 127 and the engine never starts.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/13.2.0-fasrc01 cuda/13.3.1-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

export VLLM_ENGINE_READY_TIMEOUT_S=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

:
