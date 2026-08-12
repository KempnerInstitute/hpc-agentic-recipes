# Runtime environment for DeepSeek-V4-Flash-0731 on one RTX PRO 6000 Blackwell node. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=DeepSeek-V4-Flash-0731-rtx-8
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Flash-0731/rtx-8/venv}"
[ -f "$VENV/bin/activate" ] || { echo "no environment at $VENV; run env/build.sh first" >&2; return 1 2>/dev/null || exit 1; }
export PATH="$HOME/.local/bin:$PATH"
source "$VENV/bin/activate"

CUDA13="${CUDA13_DIR:-$ENV_ROOT/DeepSeek-V4-Flash-0731/rtx-8/cuda13}"
export CUDA_HOME="$CUDA13"
[ -x "$CUDA_HOME/bin/nvcc" ] || echo "warning: no nvcc at $CUDA_HOME/bin; run env/build.sh first" >&2
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
