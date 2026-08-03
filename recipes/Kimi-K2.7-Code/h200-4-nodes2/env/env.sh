# Runtime environment for Kimi-K2.7-Code on two H200 nodes. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own. Both nodes source this
# file, so it must be valid on the Ray head and on the worker alike.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Kimi-K2.7-Code/h200-4-nodes2/venv}"
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

# required: node-local JIT caches. Concurrent compiles from two nodes against a shared NFS home hit
# stale file handles, and this recipe always compiles on two nodes at once, so a shared path is not an
# option. The cost is that the first launch on a fresh node compiles again.
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

# Load-bearing for two other checkpoints on this hardware, both verified: the DeepGEMM MoE
# path takes an illegal memory access on GLM-5.2's sparse attention, and forcing it on for
# Qwen3-Coder-480B-FP8 on H200 reproduced the same crash independently.
# inherited: the two-node run of this model used it, and it is untested for this
# model either way
export VLLM_USE_DEEP_GEMM=0

# required: weight load of 64 shards across 8 ranks plus warmup approaches the 600s default readiness
# timeout; the measured launch reached serving 9 min 11 s after the vLLM banner
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# verified: a holylfs06 OSS failover froze every rank and PyTorch's heartbeat monitor
# killed two endpoints eight minutes later, at the 480s default, for a stall that later recovered
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

# required: pin the cross-node collective and rendezvous transport to InfiniBand. These nodes carry
# both Ethernet and IB, and NCCL or Gloo picking the slow interface is the difference between a working
# two-node endpoint and one that never finishes initialization.
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
# verified: raw two-node NCCL all_reduce had to be proven healthy to establish that the
# multimodal profiling hang was not a fabric problem, and that check is only interpretable when the
# transport and its logging are pinned rather than auto-selected per run
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# List only the InfiniBand ports that are actually up. An inactive HCA left in the list stalls
# initialization, and the set of active ports is a property of the node, not of the repo, so it is
# detected here rather than hardcoded.
_hca=""
for _d in /sys/class/infiniband/*; do
  _n="$(basename "$_d")"
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  [[ "$_s" == *ACTIVE* && "$_l" == InfiniBand* ]] && _hca+="${_hca:+,}$_n"
done
# inherited: the two-node run of this model used it
export NCCL_IB_HCA="$_hca"
unset _hca _d _n _s _l

# Return success explicitly. A sourced file's status is that of its last command, and callers run under
# set -e, so ending on a failed unset or test would abort them with no output.
:
