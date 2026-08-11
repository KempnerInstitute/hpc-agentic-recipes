#!/usr/bin/env bash
# Build the environment for GLM-5.2-FP8 on two H200 nodes. Idempotent; pass --force to rebuild.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/GLM-5.2-FP8/h200-4-nodes2/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
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

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
echo "built: $VENV"
echo "the same environment must be reachable at this path from BOTH nodes; ENV_ROOT is shared scratch"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
