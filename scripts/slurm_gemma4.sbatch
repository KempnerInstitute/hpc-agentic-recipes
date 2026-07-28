#!/usr/bin/env bash
#SBATCH --job-name=gemma4-26b
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-gemma4-%j.log
#
# Gemma-4-26B-A4B-it on one RTX PRO 6000 GPU (TP1). Submit from the repo root:
#   sbatch scripts/slurm_gemma4.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/gemma-4-26B-A4B-it scripts/slurm_gemma4.sbatch
# One GPU with 16 CPUs and 180 GB is the per-GPU allowance on kempner_rtx.
# Requires the RTX environment (.venv-cu130 + cuda13); see README "RTX 6000 environment".
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$GEMMA4_MODEL}"
export ENV_LIB=lib_env_cu130.sh TP="${TP:-1}"
echo "serving Gemma-4-26B on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL)"
exec bash scripts/vllm_gemma4.sh
