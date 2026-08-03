# Runtime environment for DeepSeek-V4-Pro on two H200 nodes. Source, do not execute.
# Self-contained apart from cluster paths and key resolution: everything model or hardware specific is
# written out here rather than inherited, so this file can be read on its own.
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$S/../../../../common/defaults.sh"
KEY_NAME=DeepSeek-V4-Pro-h200-4-nodes2
source "$S/../../../../common/lib/api_key.sh"
VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/h200-4-nodes2/venv}"
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

# required: node-local JIT caches. Concurrent compiles from two nodes against a shared home hit stale
# file handles, and this is a two-node recipe, so the shared path is not an option.
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" 2>/dev/null || true

# Kept off as a counter-warning, not flipped on optimistically. Nothing about DeepSeek-V4 has been
# tested with DeepGEMM either way, and the two models that were tested on this hardware both crashed.
# verified: illegal memory access on GLM-5.2's sparse attention, and the same crash reproduced
# independently for Qwen3-Coder-480B-FP8 on H200
export VLLM_USE_DEEP_GEMM=0

# required: weight load of 806 GiB across 8 ranks plus graph work exceeds the 600s default
export VLLM_ENGINE_READY_TIMEOUT_S=3600

# verified: a holylfs06 OSS failover froze every rank and PyTorch's heartbeat monitor
# killed two endpoints eight minutes later, at the 480s default, for a stall that later recovered
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-3600}"

# inherited from the other Hopper multi-node recipes: pin the cross-node collective and rendezvous
# transport to InfiniBand, or NCCL can select a slow interface or fail to connect between nodes
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
# required: a silent NCCL failure across nodes is very hard to diagnose from the server log alone
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# List only the InfiniBand ports that are actually up, since an inactive HCA in the list stalls
# initialization.
_hca=""
for _d in /sys/class/infiniband/*; do
  _n="$(basename "$_d")"
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  [[ "$_s" == *ACTIVE* && "$_l" == InfiniBand* ]] && _hca+="${_hca:+,}$_n"
done
# inherited from the other Hopper multi-node recipes, untested for this model
export NCCL_IB_HCA="$_hca"
unset _hca _d _n _s _l
