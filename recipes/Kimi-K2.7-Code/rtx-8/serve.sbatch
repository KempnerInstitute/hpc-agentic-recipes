#!/usr/bin/env bash
#SBATCH --job-name=kimi-rtx
#SBATCH --partition=kempner_rtx
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=500G
#SBATCH --time=1-00:00:00
#SBATCH --output=%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
R=recipes/Kimi-K2.7-Code/rtx-8
echo "serving kimi-k2.7-code on $(hostname): http://$(hostname):${API_PORT:-8000}/v1"
exec bash "$R/serve.sh"
