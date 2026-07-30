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

# "Directory exists" is not "environment works": an interrupted install leaves a venv skeleton behind,
# and skipping on mere existence would then serve from a broken environment.
venv_healthy () {
  # bin/python is the proof. An interrupted install leaves site-packages populated while the
  # interpreter and activate script are missing, so a dist-info check passes on a venv that
  # cannot be activated or run.
  [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ] \
    && compgen -G "$VENV"/lib/python*/site-packages/vllm-*.dist-info > /dev/null
}
if venv_healthy && [ "${1:-}" != "--force" ]; then
  echo "environment already present and complete at $VENV (pass --force to rebuild)"
  exit 0
fi
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
[ -d "$VENV" ] && { echo "removing incomplete or forced environment"; rm -rf "$VENV" "$CUDA13"; }
mkdir -p "$(dirname "$VENV")"

# The RTX PRO 6000 is Blackwell sm_120 on a CUDA 13 driver, so this needs the CUDA 13 build of vLLM
# rather than the cu129 wheel the Hopper recipes use. Nightly wheels rotate, so record the exact
# resolution in env/requirements.lock and the resolved URLs in env/WHEELS after a successful build.
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" --prerelease=allow --index-strategy unsafe-best-match \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  vllm

# ray[default] is new for this recipe. The pre-restructure RTX environment had no Ray at all, because
# every RTX endpoint so far ran on a single node; this recipe spans two nodes, so the Ray executor is
# required. Only the Hopper environment carried Ray before.
uv pip install --python "$VENV/bin/python" "ray[default]"

# flashinfer-python must be 0.6.15, installed --no-deps so torch is left untouched. vLLM pins 0.6.13,
# whose sm_120 attention backend rejects the kv_scale_format argument vLLM passes, and that failure
# only appears at the first inference request.
uv pip install --python "$VENV/bin/python" --no-deps -U "flashinfer-python==${FLASHINFER_VERSION}"

# A complete, self-consistent CUDA 13.0 toolkit for FlashInfer's sm_120 just-in-time compilation. The
# node's /usr/local/cuda-13 is runtime-only and the fragmented pip nvcc wheels mix 13.0 with 13.2
# across nvcc, cicc, and ptxas, which breaks the JIT.
if [ ! -x "$CUDA13/bin/nvcc" ]; then
  source /etc/profile.d/lmod.sh 2>/dev/null || true
  module load Mambaforge/23.3.1-fasrc01 2>/dev/null || true
  command -v mamba >/dev/null || { echo "mamba is required for the CUDA 13.0 toolkit step" >&2; exit 1; }
  mamba create -y -p "$CUDA13" -c nvidia \
    cuda-nvcc=13.0 cuda-cudart-dev=13.0 cuda-cccl=13.0 cuda-nvrtc-dev=13.0 cuda-libraries-dev=13.0
fi

"$VENV/bin/python" -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'), '| flashinfer', m.version('flashinfer-python'))"
# Claiming success requires all three pieces, not just the venv: a build interrupted after the
# Python install leaves vllm present and the toolkit absent, which then fails at the sm_120 JIT.
# verify all three
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
