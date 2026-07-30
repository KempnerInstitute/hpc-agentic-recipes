#!/usr/bin/env bash
# Build the environment for Qwen3-Coder-480B-A35B-Instruct-FP8 on one RTX PRO 6000 node.
# Idempotent; --force rebuilds.
# This is the only supported build path: the install needs uv flags a requirements file cannot
# express, specifically --prerelease, --index-strategy, two extra index URLs and --no-deps for a
# single package, plus a conda CUDA 13.0 toolkit that no Python environment can carry.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/venv}"
CUDA13="${CUDA13_DIR:-$ENV_ROOT/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/cuda13}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.15}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
mkdir -p "$(dirname "$VENV")"

if [ -d "$VENV" ] && [ "$FORCE" = 0 ]; then
  echo "environment already present at $VENV (pass --force to rebuild)"
else
  [ "$FORCE" = 1 ] && rm -rf "$VENV"
  # The vllm wheel comes from the nightly cu130 index rather than PyPI, because sm_120 needs the
  # CUDA 13 build and uv's --torch-backend maxes out at cu129. The version is pinned explicitly:
  # the installed metadata reads a plain 0.25.1 with no local version tag, so an unpinned install
  # silently drifts, either to whatever the nightly index holds that day or to the PyPI CUDA 13
  # wheel, and neither is the build the measured numbers in the README come from.
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "vllm==$VLLM_VERSION"
  # 0.6.15 accepts the kv_scale_format argument the sm_120 attention backend passes, which the 0.6.13
  # that vLLM pins rejects at the first inference request. --no-deps keeps the resolver from ripping
  # out torch and cudnn to satisfy the newer package's own pins.
  uv pip install --python "$VENV/bin/python" --no-deps -U "flashinfer-python==$FLASHINFER_VERSION"
fi

if [ -x "$CUDA13/bin/nvcc" ] && [ "$FORCE" = 0 ]; then
  echo "CUDA 13.0 toolkit already present at $CUDA13"
else
  [ "$FORCE" = 1 ] && rm -rf "$CUDA13"
  # A complete, consistent CUDA 13.0 toolkit for FlashInfer's sm_120 JIT. The node's
  # /usr/local/cuda-13 is runtime-only, and the pip nvcc wheels mix 13.0 and 13.2 between nvcc, cicc
  # and ptxas, which surfaces as incompatible CCCL headers and then an unsupported ptx version.
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load Mambaforge/23.3.1-fasrc01 2>/dev/null || true
  command -v mamba >/dev/null || { echo "mamba is required: module load Mambaforge/23.3.1-fasrc01" >&2; exit 1; }
  mamba create -y -p "$CUDA13" -c nvidia \
    cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
fi

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| flashinfer', m.version('flashinfer-python'))"
"$CUDA13/bin/nvcc" --version | tail -1
echo "built: $VENV"
echo "       $CUDA13"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
