#!/usr/bin/env bash
# Build the environment for DeepSeek-V4-Pro on two RTX PRO 6000 nodes. Idempotent; --force rebuilds.
# This is the only supported build path: the install needs uv flags a requirements file cannot
# express, specifically a nightly index, --prerelease, --index-strategy, --no-deps for one package,
# and a conda toolkit step.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/venv}"
CUDA13="${CUDA13_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/rtx-8-nodes2/cuda13}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.15}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"

venv_healthy () {
  [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ] \
    && compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null
}
build_venv=1
if venv_healthy && [ "${1:-}" != "--force" ]; then
  echo "environment already present and complete at $VENV (pass --force to rebuild)"
  build_venv=0
fi
if [ "$build_venv" = 1 ]; then
  command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
  [ -d "$VENV" ] && { echo "removing incomplete or forced environment"; rm -rf "$VENV"; }
  mkdir -p "$(dirname "$VENV")"

  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    vllm "transformers==${TRANSFORMERS_VERSION}"

  uv pip install --python "$VENV/bin/python" "ray[default]"

  uv pip install --python "$VENV/bin/python" --no-deps -U "flashinfer-python==${FLASHINFER_VERSION}"
fi

if [ ! -x "$CUDA13/bin/nvcc" ]; then
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load Mambaforge/23.3.1-fasrc01 2>/dev/null || true
  command -v mamba >/dev/null || { echo "mamba is required for the CUDA 13.0 toolkit step" >&2; exit 1; }
  mamba create -y -p "$CUDA13" -c nvidia \
    cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
fi

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'), '| flashinfer', m.version('flashinfer-python'))"
missing=""
compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null || missing="$missing vllm"
compgen -G "$VENV"/lib/python*/site-packages/flashinfer_python-*.dist-info > /dev/null || missing="$missing flashinfer"
[ -x "$CUDA13/bin/nvcc" ] || missing="$missing cuda13-toolkit"
if [ -n "$missing" ]; then
  echo "INCOMPLETE, missing:$missing" >&2
  exit 1
fi
echo "built: $VENV"
echo "toolkit: $CUDA13"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
