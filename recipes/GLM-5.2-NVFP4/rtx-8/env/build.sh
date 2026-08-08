#!/usr/bin/env bash
# Build the environment for GLM-5.2-NVFP4 on RTX. Idempotent; pass --force to rebuild.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/GLM-5.2-NVFP4/rtx-8/venv}"
CUDA13="${CUDA13_DIR:-$ENV_ROOT/GLM-5.2-NVFP4/rtx-8/cuda13}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
FLASHINFER_VERSION="${FLASHINFER_VERSION:-0.6.15}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
mkdir -p "$(dirname "$VENV")"

venv_healthy () {
  [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ] \
    && compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null
}
if venv_healthy && [ "$FORCE" = 0 ]; then
  echo "environment already present and complete at $VENV (pass --force to rebuild)"
else
  [ -d "$VENV" ] && { echo "removing incomplete environment at $VENV"; rm -rf "$VENV"; }
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
    --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    "vllm==$VLLM_VERSION"
  uv pip install --python "$VENV/bin/python" --no-deps -U "flashinfer-python==$FLASHINFER_VERSION"
fi

if [ -x "$CUDA13/bin/nvcc" ] && [ "$FORCE" = 0 ]; then
  echo "CUDA 13.0 toolkit already present at $CUDA13"
else
  [ "$FORCE" = 1 ] && rm -rf "$CUDA13"
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load Mambaforge/23.3.1-fasrc01 2>/dev/null || true
  command -v mamba >/dev/null || { echo "mamba is required: module load Mambaforge/23.3.1-fasrc01" >&2; exit 1; }
  mamba create -y -p "$CUDA13" -c nvidia \
    cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
fi

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| flashinfer', m.version('flashinfer-python'))"
"$CUDA13/bin/nvcc" --version | tail -1
missing=""
compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null || missing="$missing vllm"
compgen -G "$VENV"/lib/python*/site-packages/flashinfer_python-*.dist-info > /dev/null || missing="$missing flashinfer"
[ -x "$CUDA13/bin/nvcc" ] || missing="$missing cuda13-toolkit"
if [ -n "$missing" ]; then
  echo "INCOMPLETE, missing:$missing" >&2
  exit 1
fi
echo "built: $VENV"
echo "       $CUDA13"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
