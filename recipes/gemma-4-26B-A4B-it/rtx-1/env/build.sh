#!/usr/bin/env bash
# Build the environment for gemma-4-26B-A4B-it on one RTX PRO 6000 Blackwell GPU.
# Idempotent; pass --force to rebuild.
# This is the only supported build path: the install needs uv flags a requirements file cannot express
# (a nightly index, --index-strategy, --prerelease, and --no-deps for a single package) plus a conda
# step for the CUDA 13.0 toolkit.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/rtx-1/venv}"
CU13="${CUDA13_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/rtx-1/cuda13}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.15}"

if [ -d "$VENV" ] && [ "${1:-}" != "--force" ]; then
  echo "environment already present at $VENV (pass --force to rebuild)"
  exit 0
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ "${1:-}" = "--force" ] && rm -rf "$VENV" "$CU13"
mkdir -p "$(dirname "$VENV")"

# sm_120 needs the CUDA 13 build of vLLM, which ships only on the nightly index. unsafe-best-match is
# what lets torch resolve from the PyTorch cu130 index while vLLM resolves from the vLLM one.
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  vllm

# flashinfer 0.6.15, not the 0.6.13 vLLM pins: the sm_120 attention backend passes a kv_scale_format
# argument that 0.6.13 rejects at the first inference request. --no-deps leaves torch untouched.
uv pip install --python "$VENV/bin/python" --no-deps -U "flashinfer-python==$FLASHINFER_VERSION"

# A complete, self-consistent CUDA 13.0 toolkit for the FlashInfer sm_120 JIT. The node's
# /usr/local/cuda-13 is runtime only, and the pip nvcc wheels mix 13.0 and 13.2 across nvcc, cicc and
# ptxas, which breaks the JIT.
if [ ! -d "$CU13" ]; then
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load Mambaforge/23.3.1-fasrc01 2>/dev/null || true
  command -v mamba >/dev/null || { echo "mamba is required for the CUDA 13.0 toolkit step" >&2; exit 1; }
  mkdir -p "$(dirname "$CU13")"
  mamba create -y -p "$CU13" -c nvidia \
    cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
fi

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| flashinfer', m.version('flashinfer-python'))"
echo "built:   $VENV"
echo "toolkit: $CU13"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
