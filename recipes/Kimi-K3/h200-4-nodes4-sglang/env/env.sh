# Runtime environment for Kimi-K3 under SGLang on four H200 nodes. Source this, do not execute.
#
# Unlike the vLLM recipes there is no virtual environment to activate: the engine ships inside a
# container and serve.sh passes each of these into it with --env. They are set here as well so the
# values live in one place and the audit can see them.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

# The container, staged beside the weights in the shared testbed so anyone who can read the checkpoint
# can run it. MODELS_DIR is checked first, so a private copy of the checkpoint can carry its own image,
# and the testbed falls back in when it does not: someone who copies only the weights to faster storage
# should not have to copy 16 GB of container as well.
_K3_TESTBED=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3/container/sglang-kimi-k3-cu12.sif
if [ -z "${SIF:-}" ]; then
  if [ -f "$MODELS_DIR/Kimi-K3/container/sglang-kimi-k3-cu12.sif" ]; then
    SIF="$MODELS_DIR/Kimi-K3/container/sglang-kimi-k3-cu12.sif"
  else
    SIF="$_K3_TESTBED"
  fi
fi

# Node-local scratch for the container's caches and logs. Nothing here may sit on a network
# filesystem: see TRITON_CACHE_DIR below.
: "${K3_LOG_DIR:=/tmp/$USER/k3}"

# required: the InfiniBand fabric is the transport for a 16-rank job, and ib0 is the interface that
# carries it on these nodes.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0

# verified: a storage stall freezes every rank, and PyTorch kills the process when the NCCL watchdog
# stops sending heartbeats, on the assumption that a collective hung. At the 480 second default a
# transient outage takes the endpoint down permanently rather than pausing it.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

# verified: Triton defaults its cache to the NFS home directory. Sixteen ranks across four nodes
# compiling the same attention metadata kernel into one shared directory raced, and capture died with
# FileNotFoundError on an entry another rank was still writing, because Triton's rename-based
# atomicity does not hold over NFS. SGLang's own hint at that point suggests memory knobs, which
# would have been the wrong fix.
export TRITON_CACHE_DIR="$K3_LOG_DIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$K3_LOG_DIR/inductor"

# required: the checkpoint is local and complete, so no rank should reach the network mid-load.
export HF_HUB_OFFLINE=1
export HF_HOME="$K3_LOG_DIR/hf"

# required: the tokenizer ships custom code and the checkpoint is bind-mounted read-only, so writing
# __pycache__ beside it would fail.
export PYTHONDONTWRITEBYTECODE=1
