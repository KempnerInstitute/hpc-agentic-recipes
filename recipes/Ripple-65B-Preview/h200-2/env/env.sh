# Runtime environment for Ripple-65B-Preview on two H200 GPUs, under SGLang in a container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_RIPPLE_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Ripple-65B-Preview/container/sglang-0.5.19.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Ripple-65B-Preview/container/sglang-0.5.19.sif" ]; then
    SIF="$MODELS_DIR/Ripple-65B-Preview/container/sglang-0.5.19.sif"
  else
    SIF="$_RIPPLE_SHARED"
  fi
fi

: "${RIPPLE_LOG_DIR:=/tmp/$USER/ripple}"

# serve.py raises this itself, but set it here too so an ssh launch cannot lose it: SGLang sizes the
# draft's positional cache for the full target context even with a windowed draft.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_HUB_OFFLINE=1
export HF_HOME="$RIPPLE_LOG_DIR/hf"
export TRITON_CACHE_DIR="$RIPPLE_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$RIPPLE_LOG_DIR/inductor"
export PYTHONDONTWRITEBYTECODE=1
