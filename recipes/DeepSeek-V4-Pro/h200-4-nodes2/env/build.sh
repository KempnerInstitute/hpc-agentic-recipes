#!/usr/bin/env bash
# Build the environment for DeepSeek-V4-Pro on two H200 nodes. Idempotent; pass --force to rebuild.
# This is the only supported build path: the install needs uv flags that a requirements file cannot
# express, specifically --torch-backend and a release wheel URL.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/DeepSeek-V4-Pro/h200-4-nodes2/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

if [ -d "$VENV" ] && [ "${1:-}" != "--force" ]; then
  echo "environment already present at $VENV (pass --force to rebuild)"
  exit 0
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ "${1:-}" = "--force" ] && rm -rf "$VENV"
mkdir -p "$(dirname "$VENV")"

# These nodes run NVIDIA driver 575 (CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel,
# so install the cu129 release wheel with a matching torch backend. ray[default] is required because
# this recipe spans two nodes and uses the Ray executor.
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --torch-backend=cu129 "$WHEEL" "ray[default]"

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
echo "built: $VENV"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
