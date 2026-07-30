#!/usr/bin/env bash
# Build the environment for gemma-4-26B-A4B-it on one H100 GPU. Idempotent; pass --force to rebuild.
# This is the only supported build path: the install needs uv flags a requirements file cannot
# express, specifically --torch-backend and a release wheel URL.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/gemma-4-26B-A4B-it/h100-1/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

# "Directory exists" is not "environment works": an interrupted install leaves a venv skeleton behind,
# and skipping on mere existence would then serve from a broken environment.
venv_healthy () { compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null; }
if venv_healthy && [ "${1:-}" != "--force" ]; then
  echo "environment already present and complete at $VENV (pass --force to rebuild)"
  exit 0
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d "$VENV" ] && { echo "removing incomplete or forced environment at $VENV"; rm -rf "$VENV"; }
mkdir -p "$(dirname "$VENV")"

# These nodes run NVIDIA driver 575 (CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel,
# so install the cu129 release wheel with a matching torch backend. Ray is not needed at TP1; it is
# installed because this is the environment the rates in the README were measured in.
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --torch-backend=cu129 "$WHEEL" "ray[default]"

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
echo "built: $VENV"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
