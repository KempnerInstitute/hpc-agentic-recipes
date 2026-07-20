# Shared runtime env for vLLM on the RTX PRO 6000 Blackwell node (sm_120, CUDA 13). Source, do not execute.
# Separate venv (.venv-cu130, torch cu130) from the H200 CUDA-12.9 envs; only runs on the RTX node.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$HOME/.local/bin:$PATH"
source "$REPO_DIR/.venv-cu130/bin/activate"
# CUDA 13.0 toolkit for FlashInfer's sm_120 sparse-MLA JIT. The node's /usr/local/cuda-13 is
# runtime-only and the fragmented pip nvcc wheels mix 13.0/13.2 (nvcc vs cicc/ptxas), which breaks
# the JIT; use the complete, consistent CUDA 13.0 toolkit installed via conda (nvidia channel).
_CU13="$REPO_DIR/cuda13"
export CUDA_HOME="${CUDA_HOME:-$_CU13}"
export PATH="$CUDA_HOME/bin:$PATH"
# Only expose the toolkit for compiling (nvcc/cicc/ptxas find their own libs via rpath).
# Do NOT prepend the toolkit's runtime libs to LD_LIBRARY_PATH: its libcudart would shadow
# torch's cu130 runtime and pull in a libcupti.so.13 that isn't present, breaking import.
# FlashInfer also compiles host C++ (tensorrt_llm cublasMMWrapper etc.) with g++, which does not
# auto-add the CUDA include/lib dirs. Expose them via CPATH/LIBRARY_PATH (compile-time only, so
# unlike LD_LIBRARY_PATH these do not shadow torch's runtime libs).
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
# CUDA 13 needs a modern host compiler; system gcc 8.5 is too old, gcc 13 is compatible.
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load gcc/13.2.0-fasrc01 2>/dev/null || true
export CUDAHOSTCXX="$(command -v g++)"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="/tmp/${USER}/triton"
export TORCHINDUCTOR_CACHE_DIR="/tmp/${USER}/torchinductor"
export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/flashinfer"
# flashinfer-python is 0.6.15 (has the kv_scale_format / fp8_ds_mla scale-format fix vLLM 0.25.1
# needs) but no matching 0.6.15 cubin package exists; bypass the version check so it JIT-compiles
# the sm_120 kernels from source (which our conda CUDA 13 toolchain handles).
export FLASHINFER_DISABLE_VERSION_CHECK=1
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_DIR" 2>/dev/null || true
# RTX PRO 6000 has no NVLink; P2P over PCIe must be disabled or NCCL init hangs.
export NCCL_P2P_DISABLE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
# API key for the served endpoint (same gitignored secrets file as the H200 models): access
# requires the key, passed via env (not argv) so it is not visible in `ps`.
_KEYFILE="$REPO_DIR/secrets/vllm_api_key"
[ -f "$_KEYFILE" ] && export VLLM_API_KEY="$(tr -d '\n\r' < "$_KEYFILE")"
