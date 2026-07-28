#!/usr/bin/env bash
#SBATCH --job-name=qwen3-235b
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-qwen3-%j.log
#
# Qwen3-235B-A22B on one full RTX PRO 6000 node (8 GPUs, TP8). Submit from the repo root:
#   sbatch scripts/slurm_qwen3.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/Qwen3-235B-A22B scripts/slurm_qwen3.sbatch
# Requires the RTX environment (.venv-cu130 + cuda13); see README "RTX 6000 environment".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$QWEN3_MODEL}"
export ENV_LIB="${ENV_LIB:-lib_env_cu130.sh}" TP="${TP:-8}"
echo "serving Qwen3-235B on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL)"
exec bash scripts/vllm_qwen3.sh
