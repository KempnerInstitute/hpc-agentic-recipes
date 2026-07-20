#!/usr/bin/env bash
#SBATCH --job-name=kimi-rtx
#SBATCH --partition=kempner_rtx
#SBATCH --account=kempner_dev
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-kimi-rtx-%j.log
#
# Kimi-K2.7-Code on one RTX PRO 6000 node (INT4, TP8). Submit from the repo root:
#   sbatch scripts/slurm_kimi_rtx.sbatch
# Override account or checkpoint:
#   sbatch --account=<acct> --export=ALL,MODEL=/path/to/Kimi-K2.7-Code scripts/slurm_kimi_rtx.sbatch
# Requires the RTX environment (.venv-cu130 + cuda13); see README "RTX 6000 environment".
# After it starts, the endpoint is http://<node>:8000/v1 on the allocated node (`squeue --me`).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
source scripts/config.sh
export MODEL="${MODEL:-$KIMI_MODEL}"
export ENV_LIB=lib_env_cu130.sh TP="${TP:-8}" ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
echo "serving Kimi-K2.7-Code on $(hostname): http://$(hostname):${API_PORT}/v1  (model: $MODEL)"
exec bash scripts/vllm_kimi.sh
