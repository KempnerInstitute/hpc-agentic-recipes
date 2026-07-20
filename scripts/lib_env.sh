# Shared runtime environment for multi-node vLLM (source this, do not execute).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$REPO_DIR/.venv/bin/activate"
# The pip CUDA stack ships no nvcc, and system gcc 8.5 is too old for DeepGEMM's C++20 kernels;
# load the cluster CUDA 12.9 + gcc 12.2 toolchain (with gmp/mpfr/mpc deps) so nvcc JIT works.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/12.2.0-fasrc01 cuda/12.9.1-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# Node-local JIT cache; shared NFS home hit stale file handles under concurrent multi-node compiles.
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
# Triton FP8 MoE path (stable). DeepGEMM MoE crashes on GLM-5.2 DSA in vLLM 0.25.1 (illegal memory access).
export VLLM_USE_DEEP_GEMM=0
# Startup (weight load + torch.compile + CUDA-graph capture) can exceed the 600s default.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export NCCL_SOCKET_IFNAME=ib0
export GLOO_SOCKET_IFNAME=ib0
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
_hca=""
for _d in /sys/class/infiniband/*; do
  _n="$(basename "$_d")"
  _s="$(cat "$_d/ports/1/state" 2>/dev/null || true)"
  _l="$(cat "$_d/ports/1/link_layer" 2>/dev/null || true)"
  [[ "$_s" == *ACTIVE* && "$_l" == InfiniBand* ]] && _hca+="${_hca:+,}$_n"
done
export NCCL_IB_HCA="$_hca"
# API key for the served endpoints (both H200 models source this): read from the gitignored
# secrets file if present, so access requires the key but it is not hardcoded in scripts or
# visible in `ps` (vLLM reads VLLM_API_KEY from the env, not argv).
_KEYFILE="$REPO_DIR/secrets/vllm_api_key"
[ -f "$_KEYFILE" ] && export VLLM_API_KEY="$(tr -d '\n\r' < "$_KEYFILE")"
