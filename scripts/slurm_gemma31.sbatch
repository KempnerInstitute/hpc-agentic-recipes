#!/usr/bin/env bash
#SBATCH --job-name=gemma4-31b
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-gemma31-%j.log
#
# Gemma-4-31B-it (dense) on one GPU, FP8 weights. Submit from the repo root:
#   sbatch scripts/slurm_gemma31.sbatch
# FP8 is the default here because it measured about 1.7x faster than bf16 on every GPU tested and
# leaves room for the full 256K context. Use QUANT= to fall back to bf16 for maximum fidelity.
# On kempner_h200 or kempner_h100 use --export=ALL,ENV_LIB=lib_env.sh and change the partition.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$GEMMA31_MODEL}"
export SERVED_NAME="${SERVED_NAME:-gemma-4-31b}"
export QUANT="${QUANT-fp8}"
export ENV_LIB="${ENV_LIB:-lib_env_cu130.sh}" TP="${TP:-1}"
echo "serving $SERVED_NAME on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL, quant ${QUANT:-bf16})"
exec bash scripts/vllm_gemma4.sh
