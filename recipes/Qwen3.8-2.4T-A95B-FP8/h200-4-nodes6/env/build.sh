#!/usr/bin/env bash
# Build the environment for Qwen3.8-2.4T-A95B-FP8 on six H200 nodes. Idempotent; pass --force to rebuild.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/Qwen3.8-2.4T-A95B-FP8/h200-4-nodes6/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.27.1}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.14.1}"

healthy () {
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

WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --torch-backend=cu129 "$WHEEL" "ray[default]" \
  "transformers==${TRANSFORMERS_VERSION}"

# torchcodec arrives as a vllm dependency and fails at import here: it needs FFmpeg 4 to 7 shared
# libraries, which this cluster does not provide. Nothing in this recipe needs it, because the model
# is text only, so it is removed rather than worked around.
uv pip uninstall --python "$VENV/bin/python" torchcodec 2>/dev/null || true

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'), '| transformers', m.version('transformers'))"
echo "built: $VENV"
echo "the same environment must be reachable at this path from ALL nodes; ENV_ROOT is shared scratch"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
