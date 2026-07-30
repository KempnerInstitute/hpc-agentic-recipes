#!/usr/bin/env bash
# Build the environment for Kimi-K2.7-Code on two H200 nodes. Idempotent; pass --force to rebuild.
# This is the only supported build path: the install needs uv flags that a requirements file cannot
# express, specifically --torch-backend and a release wheel URL.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Kimi-K2.7-Code/h200-4-nodes2/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

# "Directory exists" is not "environment works": an interrupted install leaves a venv skeleton behind,
# and skipping on mere existence would then serve from a broken environment. Check for the installed
# package instead, and treat anything else as incomplete.
healthy () {
  # bin/python is the proof. An interrupted install leaves site-packages populated while the
  # interpreter and activate script are missing, so a dist-info check passes on a venv that
  # cannot be activated or run.
  [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ] \
    && compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null
}
if [ "${1:-}" != "--force" ] && healthy; then
  echo "environment already present and complete at $VENV (pass --force to rebuild)"
  exit 0
fi
if [ -d "$VENV" ]; then
  echo "removing incomplete environment at $VENV"
  rm -rf "$VENV"
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
mkdir -p "$(dirname "$VENV")"

# These nodes run NVIDIA driver 575 (CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel,
# so install the cu129 release wheel with a matching torch backend. This is the Hopper path; the RTX
# variant of this same model needs a CUDA 13 nightly instead, which is why the two do not share an
# environment. ray[default] is required because this recipe spans two nodes through the Ray executor.
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --torch-backend=cu129 "$WHEEL" "ray[default]"

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
echo "built: $VENV"
echo "the same environment must be reachable at this path from BOTH nodes; ENV_ROOT is shared scratch"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
