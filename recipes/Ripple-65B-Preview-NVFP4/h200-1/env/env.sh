# Runtime environment for the NVFP4 build of Ripple-65B-Preview on one H200 GPU, under SGLang in a
# container.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_SIF_NAME=sglang-0.5.19.sif
_RIPPLE_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Ripple-65B-Preview-NVFP4/container/$_SIF_NAME
# One image serves both builds: the NVFP4 checkpoint imports the same SGLang layers as the BF16 one and
# pins no other version, so the BF16 sibling's copy is used rather than duplicating 12.7 GiB.
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Ripple-65B-Preview-NVFP4/container/$_SIF_NAME" ]; then
    SIF="$MODELS_DIR/Ripple-65B-Preview-NVFP4/container/$_SIF_NAME"
  elif [ -f "$MODELS_DIR/Ripple-65B-Preview/container/$_SIF_NAME" ]; then
    SIF="$MODELS_DIR/Ripple-65B-Preview/container/$_SIF_NAME"
  else
    SIF="$_RIPPLE_SHARED"
  fi
fi

: "${RIPPLE_LOG_DIR:=/tmp/$USER/ripple-nvfp4}"

# serve.py raises this itself, but set it here too so an ssh launch cannot lose it: SGLang sizes the
# draft's positional cache for the full target context even with a windowed draft.
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export HF_HUB_OFFLINE=1
export HF_HOME="$RIPPLE_LOG_DIR/hf"
export TRITON_CACHE_DIR="$RIPPLE_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$RIPPLE_LOG_DIR/inductor"
export PYTHONDONTWRITEBYTECODE=1
