#!/usr/bin/env bash
# Build the environment for MiMo-V2.6-Pro-RL on one RTX node. Idempotent; pass --force to rebuild.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/defaults.sh"
VENV="${VENV_DIR:-$ENV_ROOT/MiMo-V2.6-Pro-RL/rtx-8/venv}"
VLLM_VERSION="${VLLM_VERSION:-0.33.0}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.20.2}"

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

uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url "https://wheels.vllm.ai/${VLLM_VERSION}/cu130" \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  "vllm==$VLLM_VERSION" "ray[default]" "transformers==${TRANSFORMERS_VERSION}"

# torchcodec arrives as a vllm dependency and fails at import here: it needs FFmpeg 4 to 7 shared
# libraries, which this cluster does not provide. Nothing in this recipe needs it, because only text
# is served, so it is removed rather than worked around.
uv pip uninstall --python "$VENV/bin/python" torchcodec 2>/dev/null || true

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
echo "built: $VENV"
echo "the same environment must be reachable at this path from BOTH nodes; ENV_ROOT is shared scratch"
echo "record the exact resolution with:  uv pip freeze --python $VENV/bin/python > $S/requirements.lock"
