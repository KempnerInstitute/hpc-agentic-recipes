#!/usr/bin/env bash
#SBATCH --job-name=gemma31-rtx-1
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
#
# Submit from the repo root, and pass your account:
#   sbatch --account=<acct> recipes/gemma-4-31B-it/rtx-1/serve.sbatch
#
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/gemma-4-31B-it/rtx-1
Q="${QUANT-fp8}"
echo "serving gemma-4-31b on $(hostname): http://$(hostname):${API_PORT:-8000}/v1  (quant ${Q:-BF16})"
exec bash "$R/serve.sh"
