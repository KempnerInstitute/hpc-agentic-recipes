# Runtime environment for Muse-Spark-Open-816B-A42B-NVFP4 on one RTX node under SGLang.
# Source, do not execute.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

_MUSE_SHARED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Muse-Spark-Open-816B-A42B-NVFP4/container/sglang-muse-spark-rtxpro6000.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4/container/sglang-muse-spark-rtxpro6000.sif" ]; then
    SIF="$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4/container/sglang-muse-spark-rtxpro6000.sif"
  else
    SIF="$_MUSE_SHARED"
  fi
fi

: "${MUSE_LOG_DIR:=/tmp/$USER/muse}"

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export TRITON_CACHE_DIR="$MUSE_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$MUSE_LOG_DIR/inductor"
export HF_HUB_OFFLINE=1
export HF_HOME="$MUSE_LOG_DIR/hf"
export PYTHONDONTWRITEBYTECODE=1
