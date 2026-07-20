#!/usr/bin/env bash
# Build the uv environment with vLLM and Ray.
# The nodes run NVIDIA driver 575 (CUDA 12.9), which cannot run vLLM's default
# CUDA 13 PyPI wheel, so we install the cu129 release wheel with a cu129 torch backend.
set -euo pipefail
cd "$(dirname "$0")/.."
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"
WHEEL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python --torch-backend=cu129 "$WHEEL" "ray[default]"
.venv/bin/python -c "import importlib.metadata as m; print('vllm', m.version('vllm'), '| torch', m.version('torch'), '| ray', m.version('ray'))"
